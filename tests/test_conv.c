#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../conv.h"

static int failures;
static float scratch[8192];

static void expect_close(const char *label, float actual, float expected, float tolerance)
{
    if (fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL %s: expected %.6f, got %.6f\n", label, expected, actual);
        failures++;
    }
}

static void test_out_size(void)
{
    expect_close("out size 28, k3, pad1, s1", conv2d_out_size(28, 3, 1, 1), 28, 0);
    expect_close("out size 28, k3, pad0, s1", conv2d_out_size(28, 3, 0, 1), 26, 0);
    expect_close("out size 28, k2, pad0, s2", conv2d_out_size(28, 2, 0, 2), 14, 0);
    expect_close("out size 14, k3, pad1, s1", conv2d_out_size(14, 3, 1, 1), 14, 0);
    expect_close("out size 1, k3, pad1, s1", conv2d_out_size(1, 3, 1, 1), 1, 0);
}

static void test_forward_known(void)
{
    float input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel[4] = {1, 0, 0, -1};
    float bias[1] = {0};
    float output[4];
    float direct[4];
    float column[16];

    conv2d_forward(input, 1, 3, 3, kernel, bias, 1, 2, 0, 1, output, 2, 2, column);
    conv2d_direct_forward(input, 1, 3, 3, kernel, bias, 1, 2, 0, 1, direct, 2, 2);

    for (int i = 0; i < 4; i++) {
        expect_close("known 2x2 conv", output[i], -4.0f, 1e-5f);
        expect_close("known 2x2 direct conv", direct[i], -4.0f, 1e-5f);
    }
}

static void test_forward_padding(void)
{
    float input[1] = {5.0f};
    float kernel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    float bias[1] = {0};
    float output[9];

    conv2d_forward(input, 1, 1, 1, kernel, bias, 1, 3, 1, 1, output, 3, 3, scratch);

    /* The single pixel sits at (1, 1) of the zero-padded 3x3 image, so every
       output window covering the top-left 2x2 region sees it. */
    for (int oh = 0; oh < 3; oh++) {
        for (int ow = 0; ow < 3; ow++) {
            float expected = (oh < 2 && ow < 2) ? 5.0f : 0.0f;
            expect_close("padded 3x3 conv", output[oh * 3 + ow], expected, 1e-6f);
        }
    }
}

static void test_forward_multichannel(void)
{
    float input[8] = {1, 1, 1, 1, 2, 2, 2, 2};
    float kernel[2] = {1, -1};
    float bias[1] = {0};
    float output[4];

    conv2d_forward(input, 2, 2, 2, kernel, bias, 1, 1, 0, 1, output, 2, 2, scratch);

    for (int i = 0; i < 4; i++)
        expect_close("multi-channel conv", output[i], -1.0f, 1e-6f);
}

#define IN_C 2
#define IN_H 3
#define IN_W 3
#define OUT_C 2
#define K 2
#define OUT_H 2
#define OUT_W 2

static unsigned int rng_state = 12345u;

static float random_float(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xFFFFu) / 65535.0f - 0.5f;
}

static float conv_loss(const float *input, const float *weights, const float *biases,
                       const float *upstream)
{
    float output[OUT_C * OUT_H * OUT_W];

    conv2d_forward(input, IN_C, IN_H, IN_W, weights, biases, OUT_C, K, 0, 1,
                   output, OUT_H, OUT_W, scratch);

    float loss = 0.0f;
    for (int i = 0; i < OUT_C * OUT_H * OUT_W; i++)
        loss += output[i] * upstream[i];

    return loss;
}

static void test_backward_numeric(void)
{
    int input_count = IN_C * IN_H * IN_W;
    int weight_count = OUT_C * IN_C * K * K;

    float input[IN_C * IN_H * IN_W];
    float weights[OUT_C * IN_C * K * K];
    float biases[OUT_C];
    float upstream[OUT_C * OUT_H * OUT_W];
    float weight_grad[OUT_C * IN_C * K * K] = {0};
    float bias_grad[OUT_C] = {0};
    float input_grad[IN_C * IN_H * IN_W] = {0};

    for (int i = 0; i < input_count; i++)
        input[i] = random_float();
    for (int i = 0; i < weight_count; i++)
        weights[i] = random_float();
    for (int i = 0; i < OUT_C; i++)
        biases[i] = random_float();
    for (int i = 0; i < OUT_C * OUT_H * OUT_W; i++)
        upstream[i] = random_float();

    conv2d_backward(input, IN_C, IN_H, IN_W, weights, OUT_C, K, 0, 1,
                    upstream, OUT_H, OUT_W, weight_grad, bias_grad, input_grad, scratch);

    const float epsilon = 1e-2f;
    char label[64];

    for (int i = 0; i < weight_count; i++) {
        float original = weights[i];

        weights[i] = original + epsilon;
        float plus = conv_loss(input, weights, biases, upstream);
        weights[i] = original - epsilon;
        float minus = conv_loss(input, weights, biases, upstream);
        weights[i] = original;

        float numeric = (plus - minus) / (2.0f * epsilon);
        snprintf(label, sizeof(label), "weight grad %d", i);
        expect_close(label, weight_grad[i], numeric, 1e-2f + 1e-3f * fabsf(numeric));
    }

    for (int i = 0; i < OUT_C; i++) {
        float original = biases[i];

        biases[i] = original + epsilon;
        float plus = conv_loss(input, weights, biases, upstream);
        biases[i] = original - epsilon;
        float minus = conv_loss(input, weights, biases, upstream);
        biases[i] = original;

        float numeric = (plus - minus) / (2.0f * epsilon);
        snprintf(label, sizeof(label), "bias grad %d", i);
        expect_close(label, bias_grad[i], numeric, 1e-2f + 1e-3f * fabsf(numeric));
    }

    for (int i = 0; i < input_count; i++) {
        float original = input[i];

        input[i] = original + epsilon;
        float plus = conv_loss(input, weights, biases, upstream);
        input[i] = original - epsilon;
        float minus = conv_loss(input, weights, biases, upstream);
        input[i] = original;

        float numeric = (plus - minus) / (2.0f * epsilon);
        snprintf(label, sizeof(label), "input grad %d", i);
        expect_close(label, input_grad[i], numeric, 1e-2f + 1e-3f * fabsf(numeric));
    }
}

static void test_pool_forward(void)
{
    float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float output[4];
    int argmax[4];

    maxpool2d_forward(input, 1, 4, 4, 2, 2, output, 2, 2, argmax);

    expect_close("pool max 0", output[0], 6.0f, 0);
    expect_close("pool max 1", output[1], 8.0f, 0);
    expect_close("pool max 2", output[2], 14.0f, 0);
    expect_close("pool max 3", output[3], 16.0f, 0);
    expect_close("pool argmax 0", (float)argmax[0], 5.0f, 0);
    expect_close("pool argmax 1", (float)argmax[1], 7.0f, 0);
    expect_close("pool argmax 2", (float)argmax[2], 13.0f, 0);
    expect_close("pool argmax 3", (float)argmax[3], 15.0f, 0);
}

static void test_pool_backward(void)
{
    float input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float output[4];
    int argmax[4];
    float output_grad[4] = {1, 2, 3, 4};
    float input_grad[16];

    maxpool2d_forward(input, 1, 4, 4, 2, 2, output, 2, 2, argmax);
    maxpool2d_backward(output_grad, 1, 2, 2, argmax, 4, 4, input_grad);

    for (int i = 0; i < 16; i++) {
        float expected = 0.0f;
        if (i == 5) expected = 1.0f;
        if (i == 7) expected = 2.0f;
        if (i == 13) expected = 3.0f;
        if (i == 15) expected = 4.0f;

        char label[32];
        snprintf(label, sizeof(label), "pool backward %d", i);
        expect_close(label, input_grad[i], expected, 0);
    }
}

static void test_optimized_matches_direct(void)
{
    enum { C = 3, H = 5, W = 5, OC = 2, KK = 3, PAD = 1, OH = 5, OW = 5 };

    float input[C * H * W];
    float weights[OC * C * KK * KK];
    float biases[OC];
    float upstream[OC * OH * OW];
    float column[C * KK * KK * OH * OW];

    float out_fast[OC * OH * OW];
    float out_ref[OC * OH * OW];

    float dw_fast[OC * C * KK * KK] = {0};
    float db_fast[OC] = {0};
    float dx_fast[C * H * W] = {0};

    float dw_ref[OC * C * KK * KK] = {0};
    float db_ref[OC] = {0};
    float dx_ref[C * H * W] = {0};

    for (int i = 0; i < C * H * W; i++)
        input[i] = random_float();
    for (int i = 0; i < OC * C * KK * KK; i++)
        weights[i] = random_float();
    for (int i = 0; i < OC; i++)
        biases[i] = random_float();
    for (int i = 0; i < OC * OH * OW; i++)
        upstream[i] = random_float();

    conv2d_forward(input, C, H, W, weights, biases, OC, KK, PAD, 1,
                   out_fast, OH, OW, column);
    conv2d_direct_forward(input, C, H, W, weights, biases, OC, KK, PAD, 1,
                          out_ref, OH, OW);

    for (int i = 0; i < OC * OH * OW; i++)
        expect_close("im2col forward", out_fast[i], out_ref[i], 1e-5f);

    conv2d_backward(input, C, H, W, weights, OC, KK, PAD, 1, upstream, OH, OW,
                    dw_fast, db_fast, dx_fast, column);
    conv2d_direct_backward(input, C, H, W, weights, OC, KK, PAD, 1, upstream, OH, OW,
                           dw_ref, db_ref, dx_ref);

    for (int i = 0; i < OC * C * KK * KK; i++)
        expect_close("im2col weight grad", dw_fast[i], dw_ref[i], 1e-5f);
    for (int i = 0; i < OC; i++)
        expect_close("im2col bias grad", db_fast[i], db_ref[i], 1e-5f);
    for (int i = 0; i < C * H * W; i++)
        expect_close("im2col input grad", dx_fast[i], dx_ref[i], 1e-5f);
}

int main(void)
{
    test_out_size();
    test_forward_known();
    test_forward_padding();
    test_forward_multichannel();
    test_optimized_matches_direct();
    test_backward_numeric();
    test_pool_forward();
    test_pool_backward();

    if (failures > 0) {
        fprintf(stderr, "test_conv: %d check(s) failed\n", failures);
        return 1;
    }

    printf("test_conv: all checks passed\n");
    return 0;
}
