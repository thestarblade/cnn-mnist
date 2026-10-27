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
    layer->weight_momentum = calloc(n, sizeof(float));
    layer->bias_momentum = calloc((size_t)output_size, sizeof(float));

    for (size_t i = 0; i < n; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}

void dense_free(DenseLayer *layer)
{
    free(layer->weights);
    free(layer->biases);
    free(layer->weight_momentum);
    free(layer->bias_momentum);

    layer->weights = NULL;
    layer->biases = NULL;
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

void dense_backward(DenseLayer *layer, const float *input, const float *output_grad,
                    float *input_grad, float learning_rate, float momentum)
{
    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        if (in_j == 0.0f && !input_grad)
            continue;

        float *weight_row = &layer->weights[(size_t)j * layer->output_size];
        float *momentum_row = &layer->weight_momentum[(size_t)j * layer->output_size];
        float grad_j = 0.0f;

        for (int i = 0; i < layer->output_size; i++) {
            grad_j += output_grad[i] * weight_row[i];
            momentum_row[i] = momentum * momentum_row[i] + learning_rate * output_grad[i] * in_j;
            weight_row[i] -= momentum_row[i];
        }

        if (input_grad)
            input_grad[j] = grad_j;
    }

    for (int i = 0; i < layer->output_size; i++) {
        layer->bias_momentum[i] = momentum * layer->bias_momentum[i] + learning_rate * output_grad[i];
        layer->biases[i] -= layer->bias_momentum[i];
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
