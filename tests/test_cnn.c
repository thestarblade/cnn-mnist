#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cnn.h"

static int failures;

static void expect_close(const char *label, float actual, float expected, float tolerance)
{
    if (fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL %s: expected %.6f, got %.6f\n", label, expected, actual);
        failures++;
    }
}

static void test_shapes(void)
{
    srand(1);
    CNN net;
    cnn_init(&net);

    expect_close("conv1 out h", (float)net.conv1.out_h, (float)CNN_CONV1_H, 0);
    expect_close("conv1 out w", (float)net.conv1.out_w, (float)CNN_CONV1_W, 0);
    expect_close("pool1 out h", (float)net.pool1.out_h, (float)CNN_POOL1_H, 0);
    expect_close("conv2 out h", (float)net.conv2.out_h, (float)CNN_CONV2_H, 0);
    expect_close("pool2 out h", (float)net.pool2.out_h, (float)CNN_POOL2_H, 0);
    expect_close("flattened size", (float)CNN_FLAT_SIZE,
                 (float)(CNN_CONV2_FILTERS * 7 * 7), 0);
    expect_close("fc1 input", (float)net.fc1.input_size, (float)CNN_FLAT_SIZE, 0);
    expect_close("fc2 output", (float)net.fc2.output_size, (float)CNN_CLASSES, 0);

    cnn_free(&net);
}

static void test_zero_input(void)
{
    srand(1);
    CNN net;
    cnn_init(&net);

    float input[MNIST_PIXELS] = {0};
    cnn_forward(&net, input);

    /* Zero input kills both convolutions (biases start at zero), so every
       logit is zero and the softmax is uniform. */
    float sum = 0.0f;
    for (int i = 0; i < CNN_CLASSES; i++) {
        sum += net.probs[i];
        expect_close("uniform probability", net.probs[i], 1.0f / CNN_CLASSES, 1e-5f);
    }
    expect_close("probability sum", sum, 1.0f, 1e-5f);

    cnn_free(&net);
}

static void test_forward_is_deterministic(void)
{
    srand(7);
    CNN net;
    cnn_init(&net);

    float input[MNIST_PIXELS];
    for (int i = 0; i < MNIST_PIXELS; i++)
        input[i] = (float)((i * 37) % 256) / 255.0f;

    cnn_forward(&net, input);
    float first[CNN_CLASSES];
    memcpy(first, net.probs, sizeof(first));

    cnn_forward(&net, input);
    for (int i = 0; i < CNN_CLASSES; i++)
        expect_close("deterministic forward", net.probs[i], first[i], 0);

    cnn_free(&net);
}

static void test_activations(void)
{
    srand(11);
    CNN net;
    cnn_init(&net);

    float input[MNIST_PIXELS];
    for (int i = 0; i < MNIST_PIXELS; i++)
        input[i] = (float)((i * 11) % 256) / 255.0f;

    cnn_forward(&net, input);

    const float *relu_buffers[] = {
        net.conv1_out, net.pool1_out, net.conv2_out, net.pool2_out, net.fc1_out
    };
    const int sizes[] = {
        CNN_CONV1_UNITS, CNN_POOL1_UNITS, CNN_CONV2_UNITS, CNN_POOL2_UNITS, CNN_FC1_SIZE
    };

    for (int b = 0; b < 5; b++) {
        int max_positive = 0;
        for (int i = 0; i < sizes[b]; i++) {
            if (!isfinite(relu_buffers[b][i])) {
                fprintf(stderr, "FAIL activation buffer %d is not finite\n", b);
                failures++;
                break;
            }
            if (relu_buffers[b][i] < 0.0f) {
                fprintf(stderr, "FAIL activation buffer %d has a negative value\n", b);
                failures++;
                break;
            }
            if (relu_buffers[b][i] > 0.0f)
                max_positive = 1;
        }
        if (!max_positive) {
            fprintf(stderr, "FAIL activation buffer %d is all zeros\n", b);
            failures++;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < CNN_CLASSES; i++)
        sum += net.probs[i];
    expect_close("probability sum", sum, 1.0f, 1e-5f);

    cnn_free(&net);
}

static void test_predict_matches_argmax(void)
{
    srand(3);
    CNN net;
    cnn_init(&net);

    float input[MNIST_PIXELS];
    for (int i = 0; i < MNIST_PIXELS; i++)
        input[i] = (float)((i * 53) % 256) / 255.0f;

    cnn_forward(&net, input);

    int expected = 0;
    for (int i = 1; i < CNN_CLASSES; i++) {
        if (net.probs[i] > net.probs[expected])
            expected = i;
    }

    expect_close("predict matches argmax", (float)cnn_predict(&net, input), (float)expected, 0);

    cnn_free(&net);
}

int main(void)
{
    test_shapes();
    test_zero_input();
    test_forward_is_deterministic();
    test_activations();
    test_predict_matches_argmax();

    if (failures > 0) {
        fprintf(stderr, "test_cnn: %d check(s) failed\n", failures);
        return 1;
    }

    printf("test_cnn: all checks passed\n");
    return 0;
}
