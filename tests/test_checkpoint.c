#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../checkpoint.h"
#include "../cnn.h"

#define TEST_PATH "/tmp/cnn-checkpoint-test.cnn"

static int failures;

static void expect_close(const char *label, float actual, float expected, float tolerance)
{
    if (fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL %s: expected %.6f, got %.6f\n", label, expected, actual);
        failures++;
    }
}

static void fill_input(float *input)
{
    for (int i = 0; i < MNIST_PIXELS; i++)
        input[i] = (float)((i * 29) % 256) / 255.0f;
}

static void test_roundtrip(void)
{
    CNN saved, loaded;
    float input[MNIST_PIXELS];

    fill_input(input);

    srand(17);
    cnn_init(&saved);
    cnn_forward(&saved, input);

    float expected[CNN_CLASSES];
    memcpy(expected, saved.probs, sizeof(expected));

    if (checkpoint_save(&saved, TEST_PATH) != 0) {
        fprintf(stderr, "FAIL checkpoint_save\n");
        failures++;
    }

    srand(99);
    cnn_init(&loaded);
    cnn_forward(&loaded, input);

    int differs = 0;
    for (int i = 0; i < CNN_CLASSES; i++) {
        if (fabsf(loaded.probs[i] - expected[i]) > 1e-4f)
            differs = 1;
    }
    if (!differs) {
        fprintf(stderr, "FAIL two seeds produced the same predictions\n");
        failures++;
    }

    if (checkpoint_load(&loaded, TEST_PATH) != 0) {
        fprintf(stderr, "FAIL checkpoint_load\n");
        failures++;
    }

    cnn_forward(&loaded, input);
    for (int i = 0; i < CNN_CLASSES; i++)
        expect_close("roundtrip probability", loaded.probs[i], expected[i], 1e-6f);

    cnn_free(&saved);
    cnn_free(&loaded);
    remove(TEST_PATH);
}

static void test_missing_file(void)
{
    CNN net;

    srand(5);
    cnn_init(&net);

    if (checkpoint_load(&net, "/tmp/cnn-checkpoint-missing.cnn") == 0) {
        fprintf(stderr, "FAIL loading a missing file should fail\n");
        failures++;
    }

    cnn_free(&net);
}

static void test_rejects_garbage(void)
{
    CNN net;
    FILE *file = fopen(TEST_PATH, "wb");

    if (file) {
        fwrite("not a checkpoint", 1, 16, file);
        fclose(file);
    }

    srand(11);
    cnn_init(&net);

    if (checkpoint_load(&net, TEST_PATH) == 0) {
        fprintf(stderr, "FAIL loading a truncated file should fail\n");
        failures++;
    }

    cnn_free(&net);
    remove(TEST_PATH);
}

int main(void)
{
    test_roundtrip();
    test_missing_file();
    test_rejects_garbage();

    if (failures > 0) {
        fprintf(stderr, "test_checkpoint: %d check(s) failed\n", failures);
        return 1;
    }

    printf("test_checkpoint: all checks passed\n");
    return 0;
}
