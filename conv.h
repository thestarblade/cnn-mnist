#ifndef CONV_H
#define CONV_H

#include <stddef.h>

/* 2-D convolution over a multi-channel image.
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
 *
 * The scratch buffer must hold conv2d_col_size() floats; it stores the
 * im2col matrix and is reused for the input gradient during backward.
 */
void conv2d_forward(const float *input, int in_c, int in_h, int in_w,
                    const float *weights, const float *biases,
                    int out_c, int k, int pad, int stride,
                    float *output, int out_h, int out_w, float *col);

void conv2d_backward(const float *input, int in_c, int in_h, int in_w,
                     const float *weights, int out_c, int k, int pad, int stride,
                     const float *output_grad, int out_h, int out_w,
                     float *weight_grad, float *bias_grad, float *input_grad,
                     float *col);

size_t conv2d_col_size(int in_c, int k, int out_h, int out_w);

/* Direct implementations kept as the reference used by the tests. They need
 * no scratch buffer and are deliberately simple. */
void conv2d_direct_forward(const float *input, int in_c, int in_h, int in_w,
                           const float *weights, const float *biases,
                           int out_c, int k, int pad, int stride,
                           float *output, int out_h, int out_w);

void conv2d_direct_backward(const float *input, int in_c, int in_h, int in_w,
                            const float *weights, int out_c, int k, int pad, int stride,
                            const float *output_grad, int out_h, int out_w,
                            float *weight_grad, float *bias_grad, float *input_grad);

/* col[r * out_pixels + oh * out_w + ow] holds input[c][ih][iw] for
 * r = (c * k + ky) * k + kx, which turns the convolution into a dense
 * matrix product over r. Out-of-image positions become zero. */
void im2col(const float *input, int in_c, int in_h, int in_w,
            int k, int pad, int stride, int out_h, int out_w, float *col);

/* Accumulates col back into the input plane; input_grad must be zeroed by the
 * caller. */
void col2im(const float *col, int in_c, int in_h, int in_w,
            int k, int pad, int stride, int out_h, int out_w, float *input_grad);

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
