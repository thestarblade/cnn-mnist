#ifndef LAYER_H
#define LAYER_H

typedef struct {
    float *weights;         /* input_size * output_size, weights[j * output_size + i] */
    float *biases;          /* output_size */
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

/* Numerically stable softmax over n values, in place. */
void softmax_inplace(float *values, int n);

#endif
