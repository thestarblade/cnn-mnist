#include "conv.h"

#include <math.h>
#include <string.h>

int conv2d_out_size(int in, int k, int pad, int stride)
{
    return (in + 2 * pad - k) / stride + 1;
}

size_t conv2d_col_size(int in_c, int k, int out_h, int out_w)
{
    return (size_t)in_c * k * k * out_h * out_w;
}

void im2col(const float *input, int in_c, int in_h, int in_w,
            int k, int pad, int stride, int out_h, int out_w, float *col)
{
    int out_pixels = out_h * out_w;
    int row = 0;

    for (int c = 0; c < in_c; c++) {
        const float *in = input + (size_t)c * in_h * in_w;

        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                float *dst = col + (size_t)row * out_pixels;

                for (int oh = 0; oh < out_h; oh++) {
                    int ih = oh * stride - pad + ky;
                    float *dst_row = dst + (size_t)oh * out_w;

                    if (ih < 0 || ih >= in_h) {
                        memset(dst_row, 0, (size_t)out_w * sizeof(float));
                        continue;
                    }

                    const float *in_row = in + (size_t)ih * in_w;

                    for (int ow = 0; ow < out_w; ow++) {
                        int iw = ow * stride - pad + kx;
                        dst_row[ow] = (iw >= 0 && iw < in_w) ? in_row[iw] : 0.0f;
                    }
                }

                row++;
            }
        }
    }
}

void col2im(const float *col, int in_c, int in_h, int in_w,
            int k, int pad, int stride, int out_h, int out_w, float *input_grad)
{
    int out_pixels = out_h * out_w;
    int row = 0;

    for (int c = 0; c < in_c; c++) {
        float *in = input_grad + (size_t)c * in_h * in_w;

        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                const float *src = col + (size_t)row * out_pixels;

                for (int oh = 0; oh < out_h; oh++) {
                    int ih = oh * stride - pad + ky;
                    if (ih < 0 || ih >= in_h)
                        continue;

                    const float *src_row = src + (size_t)oh * out_w;
                    float *in_row = in + (size_t)ih * in_w;

                    for (int ow = 0; ow < out_w; ow++) {
                        int iw = ow * stride - pad + kx;
                        if (iw < 0 || iw >= in_w)
                            continue;
                        in_row[iw] += src_row[ow];
                    }
                }

                row++;
            }
        }
    }
}

void conv2d_forward(const float *input, int in_c, int in_h, int in_w,
                    const float *weights, const float *biases,
                    int out_c, int k, int pad, int stride,
                    float *output, int out_h, int out_w, float *col)
{
    int rows = in_c * k * k;
    int out_pixels = out_h * out_w;

    im2col(input, in_c, in_h, in_w, k, pad, stride, out_h, out_w, col);

    for (int oc = 0; oc < out_c; oc++) {
        const float *weights_oc = weights + (size_t)oc * rows;
        float *out = output + (size_t)oc * out_pixels;
        float bias = biases ? biases[oc] : 0.0f;

        for (int p = 0; p < out_pixels; p++)
            out[p] = bias;

        for (int r = 0; r < rows; r++) {
            float weight = weights_oc[r];
            if (weight == 0.0f)
                continue;

            const float *col_row = col + (size_t)r * out_pixels;
            for (int p = 0; p < out_pixels; p++)
                out[p] += weight * col_row[p];
        }
    }
}

void conv2d_backward(const float *input, int in_c, int in_h, int in_w,
                     const float *weights, int out_c, int k, int pad, int stride,
                     const float *output_grad, int out_h, int out_w,
                     float *weight_grad, float *bias_grad, float *input_grad,
                     float *col)
{
    int rows = in_c * k * k;
    int out_pixels = out_h * out_w;

    im2col(input, in_c, in_h, in_w, k, pad, stride, out_h, out_w, col);

    for (int oc = 0; oc < out_c; oc++) {
        const float *dout = output_grad + (size_t)oc * out_pixels;
        float *dw = weight_grad ? weight_grad + (size_t)oc * rows : NULL;

        if (bias_grad) {
            float sum = 0.0f;
            for (int p = 0; p < out_pixels; p++)
                sum += dout[p];
            bias_grad[oc] += sum;
        }

        if (!dw)
            continue;

        for (int r = 0; r < rows; r++) {
            const float *col_row = col + (size_t)r * out_pixels;
            float grad = 0.0f;

            for (int p = 0; p < out_pixels; p++)
                grad += dout[p] * col_row[p];

            dw[r] += grad;
        }
    }

    if (!input_grad)
        return;

    /* Reuse the im2col buffer for dcol = W^T * dout, then scatter it back. */
    for (int r = 0; r < rows; r++) {
        float *dst = col + (size_t)r * out_pixels;
        memset(dst, 0, (size_t)out_pixels * sizeof(float));

        for (int oc = 0; oc < out_c; oc++) {
            float weight = weights[(size_t)oc * rows + r];
            const float *dout = output_grad + (size_t)oc * out_pixels;

            for (int p = 0; p < out_pixels; p++)
                dst[p] += weight * dout[p];
        }
    }

    memset(input_grad, 0, (size_t)in_c * in_h * in_w * sizeof(float));
    col2im(col, in_c, in_h, in_w, k, pad, stride, out_h, out_w, input_grad);
}

void conv2d_direct_forward(const float *input, int in_c, int in_h, int in_w,
                           const float *weights, const float *biases,
                           int out_c, int k, int pad, int stride,
                           float *output, int out_h, int out_w)
{
    int out_pixels = out_h * out_w;

    for (int oc = 0; oc < out_c; oc++) {
        float *out = output + (size_t)oc * out_pixels;
        const float *weights_oc = weights + (size_t)oc * in_c * k * k;
        float bias = biases ? biases[oc] : 0.0f;

        for (int p = 0; p < out_pixels; p++)
            out[p] = bias;

        for (int ic = 0; ic < in_c; ic++) {
            const float *in = input + (size_t)ic * in_h * in_w;
            const float *weights_ic = weights_oc + (size_t)ic * k * k;

            for (int ky = 0; ky < k; ky++) {
                for (int kx = 0; kx < k; kx++) {
                    float weight = weights_ic[ky * k + kx];
                    if (weight == 0.0f)
                        continue;

                    for (int oh = 0; oh < out_h; oh++) {
                        int ih = oh * stride - pad + ky;
                        if (ih < 0 || ih >= in_h)
                            continue;

                        const float *in_row = in + (size_t)ih * in_w;
                        float *out_row = out + (size_t)oh * out_w;

                        for (int ow = 0; ow < out_w; ow++) {
                            int iw = ow * stride - pad + kx;
                            if (iw < 0 || iw >= in_w)
                                continue;
                            out_row[ow] += weight * in_row[iw];
                        }
                    }
                }
            }
        }
    }
}

void conv2d_direct_backward(const float *input, int in_c, int in_h, int in_w,
                            const float *weights, int out_c, int k, int pad, int stride,
                            const float *output_grad, int out_h, int out_w,
                            float *weight_grad, float *bias_grad, float *input_grad)
{
    int out_pixels = out_h * out_w;

    if (input_grad)
        memset(input_grad, 0, (size_t)in_c * in_h * in_w * sizeof(float));

    for (int oc = 0; oc < out_c; oc++) {
        const float *dout = output_grad + (size_t)oc * out_pixels;
        float *dw_oc = weight_grad ? weight_grad + (size_t)oc * in_c * k * k : NULL;

        if (bias_grad) {
            float sum = 0.0f;
            for (int p = 0; p < out_pixels; p++)
                sum += dout[p];
            bias_grad[oc] += sum;
        }

        for (int ic = 0; ic < in_c; ic++) {
            const float *in = input + (size_t)ic * in_h * in_w;
            const float *weights_ic = weights + (size_t)oc * in_c * k * k + (size_t)ic * k * k;
            float *dw_ic = dw_oc ? dw_oc + (size_t)ic * k * k : NULL;
            float *din = input_grad ? input_grad + (size_t)ic * in_h * in_w : NULL;

            for (int ky = 0; ky < k; ky++) {
                for (int kx = 0; kx < k; kx++) {
                    float weight = weights_ic[ky * k + kx];
                    float grad = 0.0f;

                    for (int oh = 0; oh < out_h; oh++) {
                        int ih = oh * stride - pad + ky;
                        if (ih < 0 || ih >= in_h)
                            continue;

                        const float *in_row = in + (size_t)ih * in_w;
                        const float *dout_row = dout + (size_t)oh * out_w;
                        float *din_row = din ? din + (size_t)ih * in_w : NULL;

                        for (int ow = 0; ow < out_w; ow++) {
                            int iw = ow * stride - pad + kx;
                            if (iw < 0 || iw >= in_w)
                                continue;

                            float d = dout_row[ow];
                            grad += d * in_row[iw];
                            if (din_row)
                                din_row[iw] += weight * d;
                        }
                    }

                    if (dw_ic)
                        dw_ic[ky * k + kx] += grad;
                }
            }
        }
    }
}

void maxpool2d_forward(const float *input, int channels, int in_h, int in_w,
                       int size, int stride, float *output, int out_h, int out_w,
                       int *argmax)
{
    int out_pixels = out_h * out_w;

    for (int c = 0; c < channels; c++) {
        const float *in = input + (size_t)c * in_h * in_w;
        float *out = output + (size_t)c * out_pixels;
        int *index = argmax ? argmax + (size_t)c * out_pixels : NULL;

        for (int oh = 0; oh < out_h; oh++) {
            int base_h = oh * stride;

            for (int ow = 0; ow < out_w; ow++) {
                int base_w = ow * stride;
                float best = -INFINITY;
                int best_index = base_h * in_w + base_w;

                for (int kh = 0; kh < size; kh++) {
                    int ih = base_h + kh;
                    if (ih >= in_h)
                        break;

                    for (int kw = 0; kw < size; kw++) {
                        int iw = base_w + kw;
                        if (iw >= in_w)
                            break;

                        float value = in[ih * in_w + iw];
                        if (value > best) {
                            best = value;
                            best_index = ih * in_w + iw;
                        }
                    }
                }

                out[oh * out_w + ow] = best;
                if (index)
                    index[oh * out_w + ow] = best_index;
            }
        }
    }
}

void maxpool2d_backward(const float *output_grad, int channels, int out_h, int out_w,
                        const int *argmax, int in_h, int in_w, float *input_grad)
{
    int out_pixels = out_h * out_w;

    memset(input_grad, 0, (size_t)channels * in_h * in_w * sizeof(float));

    for (int c = 0; c < channels; c++) {
        const float *dout = output_grad + (size_t)c * out_pixels;
        const int *index = argmax + (size_t)c * out_pixels;
        float *din = input_grad + (size_t)c * in_h * in_w;

        for (int p = 0; p < out_pixels; p++)
            din[index[p]] += dout[p];
    }
}
