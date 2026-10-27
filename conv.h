#ifndef CONV_H
#define CONV_H

/* 2-D convolution over a multi-channel image, direct implementation.
 *
 * Shapes are channel-major:
 *   input   in_c  x in_h  x in_w
 *   weights out_c x in_c  x k  x k
 *   output  out_c x out_h x out_w
 *
 * No padding value is added on the border beyond zeros, so pad must be
 * smaller than k. input_grad, bias_grad and weight_grad may be NULL when
 * the corresponding gradient is not needed; weight_grad and bias_grad are
 * accumulated, not overwritten.
 */
void conv2d_forward(const float *input, int in_c, int in_h, int in_w,
                    const float *weights, const float *biases,
                    int out_c, int k, int pad, int stride,
                    float *output, int out_h, int out_w);

void conv2d_backward(const float *input, int in_c, int in_h, int in_w,
                     const float *weights, int out_c, int k, int pad, int stride,
                     const float *output_grad, int out_h, int out_w,
                     float *weight_grad, float *bias_grad, float *input_grad);

/* 2x2 style max pooling without padding. argmax stores, for every output
 * position, the index of the winning element inside its input channel, which
 * is all maxpool2d_backward needs to route the gradient. */
void maxpool2d_forward(const float *input, int channels, int in_h, int in_w,
                       int size, int stride, float *output, int out_h, int out_w,
                       int *argmax);

void maxpool2d_backward(const float *output_grad, int channels, int out_h, int out_w,
                        const int *argmax, int in_h, int in_w, float *input_grad);

int conv2d_out_size(int in, int k, int pad, int stride);

#endif
