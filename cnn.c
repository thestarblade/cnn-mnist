#include "cnn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "conv.h"

static float random_weight(float fan_in)
{
    float scale = sqrtf(2.0f / fan_in);
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
}

static void conv_init(ConvLayer *layer, int in_c, int in_h, int in_w,
                      int out_c, int k, int pad, int stride)
{
    size_t count = (size_t)out_c * in_c * k * k;

    layer->in_c = in_c;
    layer->in_h = in_h;
    layer->in_w = in_w;
    layer->out_c = out_c;
    layer->out_h = conv2d_out_size(in_h, k, pad, stride);
    layer->out_w = conv2d_out_size(in_w, k, pad, stride);
    layer->k = k;
    layer->pad = pad;
    layer->stride = stride;

    layer->weights = malloc(count * sizeof(float));
    layer->biases = calloc((size_t)out_c, sizeof(float));
    layer->weight_grad = calloc(count, sizeof(float));
    layer->bias_grad = calloc((size_t)out_c, sizeof(float));
    layer->weight_momentum = calloc(count, sizeof(float));
    layer->bias_momentum = calloc((size_t)out_c, sizeof(float));

    float fan_in = (float)(in_c * k * k);
    for (size_t i = 0; i < count; i++)
        layer->weights[i] = random_weight(fan_in);
}

static void conv_free(ConvLayer *layer)
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

static void pool_init(PoolLayer *layer, int channels, int in_h, int in_w, int size, int stride)
{
    layer->channels = channels;
    layer->in_h = in_h;
    layer->in_w = in_w;
    layer->out_h = (in_h - size) / stride + 1;
    layer->out_w = (in_w - size) / stride + 1;
    layer->size = size;
    layer->stride = stride;
    layer->argmax = malloc((size_t)channels * layer->out_h * layer->out_w * sizeof(int));
}

static void pool_free(PoolLayer *layer)
{
    free(layer->argmax);
    layer->argmax = NULL;
}

static void relu_inplace(float *values, int n)
{
    for (int i = 0; i < n; i++) {
        if (values[i] < 0.0f)
            values[i] = 0.0f;
    }
}

void cnn_init(CNN *net)
{
    conv_init(&net->conv1, 1, MNIST_ROWS, MNIST_COLS, CNN_CONV1_FILTERS,
              CNN_KERNEL, CNN_PADDING, 1);
    pool_init(&net->pool1, CNN_CONV1_FILTERS, CNN_CONV1_H, CNN_CONV1_W, CNN_POOL, CNN_POOL);
    conv_init(&net->conv2, CNN_CONV1_FILTERS, CNN_POOL1_H, CNN_POOL1_W, CNN_CONV2_FILTERS,
              CNN_KERNEL, CNN_PADDING, 1);
    pool_init(&net->pool2, CNN_CONV2_FILTERS, CNN_CONV2_H, CNN_CONV2_W, CNN_POOL, CNN_POOL);

    dense_init(&net->fc1, CNN_FLAT_SIZE, CNN_FC1_SIZE);
    dense_init(&net->fc2, CNN_FC1_SIZE, CNN_CLASSES);

    net->input = malloc(MNIST_PIXELS * sizeof(float));
    net->conv1_out = malloc(CNN_CONV1_UNITS * sizeof(float));
    net->pool1_out = malloc(CNN_POOL1_UNITS * sizeof(float));
    net->conv2_out = malloc(CNN_CONV2_UNITS * sizeof(float));
    net->pool2_out = malloc(CNN_POOL2_UNITS * sizeof(float));
    net->fc1_out = malloc(CNN_FC1_SIZE * sizeof(float));
    net->logits = malloc(CNN_CLASSES * sizeof(float));
    net->probs = malloc(CNN_CLASSES * sizeof(float));

    net->d_conv1_out = malloc(CNN_CONV1_UNITS * sizeof(float));
    net->d_pool1_out = malloc(CNN_POOL1_UNITS * sizeof(float));
    net->d_conv2_out = malloc(CNN_CONV2_UNITS * sizeof(float));
    net->d_pool2_out = malloc(CNN_POOL2_UNITS * sizeof(float));
    net->d_fc1_out = malloc(CNN_FC1_SIZE * sizeof(float));
}

void cnn_free(CNN *net)
{
    conv_free(&net->conv1);
    pool_free(&net->pool1);
    conv_free(&net->conv2);
    pool_free(&net->pool2);
    dense_free(&net->fc1);
    dense_free(&net->fc2);

    free(net->input);
    free(net->conv1_out);
    free(net->pool1_out);
    free(net->conv2_out);
    free(net->pool2_out);
    free(net->fc1_out);
    free(net->logits);
    free(net->probs);
    free(net->d_conv1_out);
    free(net->d_pool1_out);
    free(net->d_conv2_out);
    free(net->d_pool2_out);
    free(net->d_fc1_out);

    net->input = NULL;
    net->conv1_out = NULL;
    net->pool1_out = NULL;
    net->conv2_out = NULL;
    net->pool2_out = NULL;
    net->fc1_out = NULL;
    net->logits = NULL;
    net->probs = NULL;
    net->d_conv1_out = NULL;
    net->d_pool1_out = NULL;
    net->d_conv2_out = NULL;
    net->d_pool2_out = NULL;
    net->d_fc1_out = NULL;
}

void cnn_forward(CNN *net, const float *input)
{
    memcpy(net->input, input, MNIST_PIXELS * sizeof(float));

    conv2d_forward(net->input, 1, MNIST_ROWS, MNIST_COLS,
                   net->conv1.weights, net->conv1.biases,
                   CNN_CONV1_FILTERS, CNN_KERNEL, CNN_PADDING, 1,
                   net->conv1_out, CNN_CONV1_H, CNN_CONV1_W);
    relu_inplace(net->conv1_out, CNN_CONV1_UNITS);

    maxpool2d_forward(net->conv1_out, CNN_CONV1_FILTERS, CNN_CONV1_H, CNN_CONV1_W,
                      CNN_POOL, CNN_POOL, net->pool1_out, CNN_POOL1_H, CNN_POOL1_W,
                      net->pool1.argmax);

    conv2d_forward(net->pool1_out, CNN_CONV1_FILTERS, CNN_POOL1_H, CNN_POOL1_W,
                   net->conv2.weights, net->conv2.biases,
                   CNN_CONV2_FILTERS, CNN_KERNEL, CNN_PADDING, 1,
                   net->conv2_out, CNN_CONV2_H, CNN_CONV2_W);
    relu_inplace(net->conv2_out, CNN_CONV2_UNITS);

    maxpool2d_forward(net->conv2_out, CNN_CONV2_FILTERS, CNN_CONV2_H, CNN_CONV2_W,
                      CNN_POOL, CNN_POOL, net->pool2_out, CNN_POOL2_H, CNN_POOL2_W,
                      net->pool2.argmax);

    dense_forward(&net->fc1, net->pool2_out, net->fc1_out, 1);
    dense_forward(&net->fc2, net->fc1_out, net->logits, 0);

    memcpy(net->probs, net->logits, CNN_CLASSES * sizeof(float));
    softmax_inplace(net->probs, CNN_CLASSES);
}

int cnn_predict(CNN *net, const float *input)
{
    cnn_forward(net, input);

    int best = 0;
    for (int i = 1; i < CNN_CLASSES; i++) {
        if (net->probs[i] > net->probs[best])
            best = i;
    }

    return best;
}

void cnn_zero_grad(CNN *net)
{
    size_t conv1_weights = (size_t)CNN_CONV1_FILTERS * 1 * CNN_KERNEL * CNN_KERNEL;
    size_t conv2_weights = (size_t)CNN_CONV2_FILTERS * CNN_CONV1_FILTERS * CNN_KERNEL * CNN_KERNEL;

    memset(net->conv1.weight_grad, 0, conv1_weights * sizeof(float));
    memset(net->conv1.bias_grad, 0, CNN_CONV1_FILTERS * sizeof(float));
    memset(net->conv2.weight_grad, 0, conv2_weights * sizeof(float));
    memset(net->conv2.bias_grad, 0, CNN_CONV2_FILTERS * sizeof(float));

    dense_zero_grad(&net->fc1);
    dense_zero_grad(&net->fc2);
}

void cnn_backward(CNN *net, int label)
{
    float d_logits[CNN_CLASSES];

    /* Softmax + cross-entropy simplifies to probs - onehot. */
    for (int i = 0; i < CNN_CLASSES; i++)
        d_logits[i] = net->probs[i] - (i == label ? 1.0f : 0.0f);

    dense_backward_accumulate(&net->fc2, net->fc1_out, d_logits, net->d_fc1_out);
    relu_backward(net->d_fc1_out, net->fc1_out, CNN_FC1_SIZE);

    dense_backward_accumulate(&net->fc1, net->pool2_out, net->d_fc1_out, net->d_pool2_out);

    maxpool2d_backward(net->d_pool2_out, CNN_CONV2_FILTERS, CNN_POOL2_H, CNN_POOL2_W,
                       net->pool2.argmax, CNN_CONV2_H, CNN_CONV2_W, net->d_conv2_out);
    relu_backward(net->d_conv2_out, net->conv2_out, CNN_CONV2_UNITS);

    conv2d_backward(net->pool1_out, CNN_CONV1_FILTERS, CNN_POOL1_H, CNN_POOL1_W,
                    net->conv2.weights, CNN_CONV2_FILTERS, CNN_KERNEL, CNN_PADDING, 1,
                    net->d_conv2_out, CNN_CONV2_H, CNN_CONV2_W,
                    net->conv2.weight_grad, net->conv2.bias_grad, net->d_pool1_out);

    maxpool2d_backward(net->d_pool1_out, CNN_CONV1_FILTERS, CNN_POOL1_H, CNN_POOL1_W,
                       net->pool1.argmax, CNN_CONV1_H, CNN_CONV1_W, net->d_conv1_out);
    relu_backward(net->d_conv1_out, net->conv1_out, CNN_CONV1_UNITS);

    conv2d_backward(net->input, 1, MNIST_ROWS, MNIST_COLS,
                    net->conv1.weights, CNN_CONV1_FILTERS, CNN_KERNEL, CNN_PADDING, 1,
                    net->d_conv1_out, CNN_CONV1_H, CNN_CONV1_W,
                    net->conv1.weight_grad, net->conv1.bias_grad, NULL);
}

static void conv_step(ConvLayer *layer, float learning_rate, float momentum)
{
    size_t count = (size_t)layer->out_c * layer->in_c * layer->k * layer->k;

    for (size_t i = 0; i < count; i++) {
        float update = momentum * layer->weight_momentum[i] + learning_rate * layer->weight_grad[i];
        layer->weight_momentum[i] = update;
        layer->weights[i] -= update;
        layer->weight_grad[i] = 0.0f;
    }

    for (int i = 0; i < layer->out_c; i++) {
        float update = momentum * layer->bias_momentum[i] + learning_rate * layer->bias_grad[i];
        layer->bias_momentum[i] = update;
        layer->biases[i] -= update;
        layer->bias_grad[i] = 0.0f;
    }
}

void cnn_step(CNN *net, float learning_rate, float momentum)
{
    conv_step(&net->conv1, learning_rate, momentum);
    conv_step(&net->conv2, learning_rate, momentum);
    dense_step(&net->fc1, learning_rate, momentum);
    dense_step(&net->fc2, learning_rate, momentum);
}
