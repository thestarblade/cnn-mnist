#ifndef LAYER_H
#define LAYER_H

typedef struct {
    float *weights;         /* input_size * output_size, weights[j * output_size + i] */
    float *biases;          /* output_size */
    float *weight_grad;     /* same shape as weights */
    float *bias_grad;       /* output_size */
    float *weight_momentum; /* same shape as weights */
    float *bias_momentum;   /* output_size */
    int input_size;
    int output_size;
} DenseLayer;

void dense_init(DenseLayer *layer, int input_size, int output_size);
void dense_free(DenseLayer *layer);

void dense_forward(const DenseLayer *layer, const float *input, float *output, int relu);

/* Accumulates gradients for one sample into weight_grad/bias_grad, so a caller
 * can sum them over a mini-batch. Use dense_zero_grad() before the first
 * sample and dense_step() once the batch is complete. */
void dense_backward_accumulate(DenseLayer *layer, const float *input,
                               const float *output_grad, float *input_grad);

void dense_zero_grad(DenseLayer *layer);

/* Applies the accumulated gradients with momentum SGD and clears them. */
void dense_step(DenseLayer *layer, float learning_rate, float momentum);

/* Numerically stable softmax over n values, in place. */
void softmax_inplace(float *values, int n);

/* Multiplies a gradient by the ReLU derivative of an activation buffer. */
void relu_backward(float *grad, const float *activation, int n);

#endif
