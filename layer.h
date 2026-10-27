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

/* Backpropagates one sample and applies the momentum update in the same pass.
 * output_grad must already include the activation derivative. input_grad may
 * be NULL when the input gradient is not needed. */
void dense_backward(DenseLayer *layer, const float *input, const float *output_grad,
                    float *input_grad, float learning_rate, float momentum);

/* Same as dense_backward but only accumulates into weight_grad/bias_grad, so a
 * caller can sum gradients over a mini-batch. Use dense_zero_grad() before the
 * first sample and dense_step() once the batch is complete. */
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
