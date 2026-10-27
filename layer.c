#include "layer.h"

#include <math.h>
#include <stdlib.h>

void dense_init(DenseLayer *layer, int input_size, int output_size)
{
    size_t n = (size_t)input_size * (size_t)output_size;
    float scale = sqrtf(2.0f / (float)input_size);

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc((size_t)output_size, sizeof(float));
    layer->weight_grad = calloc(n, sizeof(float));
    layer->bias_grad = calloc((size_t)output_size, sizeof(float));
    layer->weight_momentum = calloc(n, sizeof(float));
    layer->bias_momentum = calloc((size_t)output_size, sizeof(float));

    for (size_t i = 0; i < n; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}

void dense_free(DenseLayer *layer)
{
    free(layer->weights);
    free(layer->biases);
    free(layer->weight_grad);
    free(layer->bias_grad);
    free(layer->weight_momentum);
    free(layer->bias_momentum);

    layer->weights = NULL;
    layer->biases = NULL;
    layer->weight_grad = NULL;
    layer->bias_grad = NULL;
    layer->weight_momentum = NULL;
    layer->bias_momentum = NULL;
}

void dense_forward(const DenseLayer *layer, const float *input, float *output, int relu)
{
    for (int i = 0; i < layer->output_size; i++)
        output[i] = layer->biases[i];

    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        if (in_j == 0.0f)
            continue;

        const float *weight_row = &layer->weights[(size_t)j * layer->output_size];
        for (int i = 0; i < layer->output_size; i++)
            output[i] += in_j * weight_row[i];
    }

    if (relu) {
        for (int i = 0; i < layer->output_size; i++)
            output[i] = output[i] > 0.0f ? output[i] : 0.0f;
    }
}

void dense_backward_accumulate(DenseLayer *layer, const float *input,
                               const float *output_grad, float *input_grad)
{
    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        const float *weight_row = &layer->weights[(size_t)j * layer->output_size];
        float grad_j = 0.0f;

        if (in_j != 0.0f) {
            float *grad_row = &layer->weight_grad[(size_t)j * layer->output_size];
            for (int i = 0; i < layer->output_size; i++)
                grad_row[i] += output_grad[i] * in_j;
        }

        if (input_grad) {
            for (int i = 0; i < layer->output_size; i++)
                grad_j += output_grad[i] * weight_row[i];
            input_grad[j] = grad_j;
        }
    }

    for (int i = 0; i < layer->output_size; i++)
        layer->bias_grad[i] += output_grad[i];
}

void dense_zero_grad(DenseLayer *layer)
{
    size_t n = (size_t)layer->input_size * (size_t)layer->output_size;

    for (size_t i = 0; i < n; i++)
        layer->weight_grad[i] = 0.0f;

    for (int i = 0; i < layer->output_size; i++)
        layer->bias_grad[i] = 0.0f;
}

void dense_step(DenseLayer *layer, float learning_rate, float momentum)
{
    size_t n = (size_t)layer->input_size * (size_t)layer->output_size;

    for (size_t i = 0; i < n; i++) {
        float update = momentum * layer->weight_momentum[i] + learning_rate * layer->weight_grad[i];
        layer->weight_momentum[i] = update;
        layer->weights[i] -= update;
        layer->weight_grad[i] = 0.0f;
    }

    for (int i = 0; i < layer->output_size; i++) {
        float update = momentum * layer->bias_momentum[i] + learning_rate * layer->bias_grad[i];
        layer->bias_momentum[i] = update;
        layer->biases[i] -= update;
        layer->bias_grad[i] = 0.0f;
    }
}

void softmax_inplace(float *values, int n)
{
    float max = values[0];
    for (int i = 1; i < n; i++) {
        if (values[i] > max)
            max = values[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        values[i] = expf(values[i] - max);
        sum += values[i];
    }

    for (int i = 0; i < n; i++)
        values[i] /= sum;
}

void relu_backward(float *grad, const float *activation, int n)
{
    for (int i = 0; i < n; i++) {
        if (activation[i] <= 0.0f)
            grad[i] = 0.0f;
    }
}
