#ifndef CNN_H
#define CNN_H

#include "layer.h"
#include "mnist.h"

#define CNN_CONV1_FILTERS 8
#define CNN_CONV2_FILTERS 16
#define CNN_KERNEL 3
#define CNN_PADDING 1
#define CNN_POOL 2
#define CNN_FC1_SIZE 64
#define CNN_CLASSES 10

/* Spatial sizes derived from the fixed MNIST layout:
   28x28 -> conv1 (8x28x28) -> pool1 (8x14x14)
         -> conv2 (16x14x14) -> pool2 (16x7x7) -> fc1 (64) -> fc2 (10) */
#define CNN_CONV1_H MNIST_ROWS
#define CNN_CONV1_W MNIST_COLS
#define CNN_POOL1_H (CNN_CONV1_H / CNN_POOL)
#define CNN_POOL1_W (CNN_CONV1_W / CNN_POOL)
#define CNN_CONV2_H CNN_POOL1_H
#define CNN_CONV2_W CNN_POOL1_W
#define CNN_POOL2_H (CNN_CONV2_H / CNN_POOL)
#define CNN_POOL2_W (CNN_CONV2_W / CNN_POOL)
#define CNN_FLAT_SIZE (CNN_CONV2_FILTERS * CNN_POOL2_H * CNN_POOL2_W)

#define CNN_CONV1_UNITS (CNN_CONV1_FILTERS * CNN_CONV1_H * CNN_CONV1_W)
#define CNN_POOL1_UNITS (CNN_CONV1_FILTERS * CNN_POOL1_H * CNN_POOL1_W)
#define CNN_CONV2_UNITS (CNN_CONV2_FILTERS * CNN_CONV2_H * CNN_CONV2_W)
#define CNN_POOL2_UNITS (CNN_CONV2_FILTERS * CNN_POOL2_H * CNN_POOL2_W)

typedef struct {
    float *weights;    /* out_c x in_c x k x k */
    float *biases;     /* out_c */
    float *weight_grad; /* same shape as weights, accumulated */
    float *bias_grad;   /* out_c, accumulated */
    float *weight_momentum; /* same shape as weights */
    float *bias_momentum;   /* out_c */
    int in_c, in_h, in_w;
    int out_c, out_h, out_w;
    int k, pad, stride;
} ConvLayer;

typedef struct {
    int channels;
    int in_h, in_w;
    int out_h, out_w;
    int size, stride;
    int *argmax; /* channels x out_h x out_w indices into the input channel */
} PoolLayer;

typedef struct {
    ConvLayer conv1;
    PoolLayer pool1;
    ConvLayer conv2;
    PoolLayer pool2;
    DenseLayer fc1;
    DenseLayer fc2;

    float *input; /* normalized copy of the 28x28 input */
    float *conv1_out;
    float *pool1_out;
    float *conv2_out;
    float *pool2_out;
    float *fc1_out;
    float *logits;
    float *probs;

    /* Gradients w.r.t. the activations, reused by every backward pass. */
    float *d_conv1_out;
    float *d_pool1_out;
    float *d_conv2_out;
    float *d_pool2_out;
    float *d_fc1_out;
} CNN;

void cnn_init(CNN *net);
void cnn_free(CNN *net);

/* Runs the network on MNIST_PIXELS values in [0, 1] and fills net->probs. */
void cnn_forward(CNN *net, const float *input);

int cnn_predict(CNN *net, const float *input);

/* Must be called after cnn_forward() on the same sample. Gradients are
 * accumulated into the layer buffers, so zero them once per mini-batch. */
void cnn_backward(CNN *net, int label);

void cnn_zero_grad(CNN *net);

/* Applies the accumulated gradients with momentum SGD and then clears them. */
void cnn_step(CNN *net, float learning_rate, float momentum);

#endif
