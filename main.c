#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cnn.h"
#include "mnist.h"

#ifndef EPOCHS
#define EPOCHS 10
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 32
#endif
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01f
#endif
#ifndef MOMENTUM
#define MOMENTUM 0.9f
#endif
#ifndef TRAIN_SPLIT
#define TRAIN_SPLIT 0.8
#endif
#ifndef SEED
#define SEED 0
#endif

#define DEFAULT_DATA_DIR "data"
#define PATH_LENGTH 512

typedef struct {
    int epochs;
    int batch_size;
    float learning_rate;
    float momentum;
    unsigned int seed;
    const char *data_dir;
} Options;

static void print_usage(const char *program)
{
    printf("Usage: %s [options]\n"
           "\n"
           "Options:\n"
           "  --epochs N        number of epochs (default %d, 0 = evaluate only)\n"
           "  --batch-size N    samples per gradient step (default %d)\n"
           "  --lr F            learning rate (default %.4f)\n"
           "  --momentum F      SGD momentum (default %.2f)\n"
           "  --seed N          random seed (default: current time)\n"
           "  --data-dir DIR    directory holding the IDX files (default %s)\n"
           "  -h, --help        show this help\n",
           program, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, DEFAULT_DATA_DIR);
}

static const char *option_value(int argc, char **argv, int *index, const char *name)
{
    if (*index + 1 >= argc) {
        fprintf(stderr, "error: %s requires a value\n", name);
        return NULL;
    }

    (*index)++;
    return argv[*index];
}

static int parse_int(const char *text, int *value)
{
    char *end;
    long parsed = strtol(text, &end, 10);

    if (*text == '\0' || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX)
        return -1;

    *value = (int)parsed;
    return 0;
}

static int parse_float(const char *text, float *value)
{
    char *end;
    float parsed = strtof(text, &end);

    if (*text == '\0' || *end != '\0' || !isfinite(parsed))
        return -1;

    *value = parsed;
    return 0;
}

static int parse_options(int argc, char **argv, Options *options)
{
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *value;

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--epochs") == 0) {
            value = option_value(argc, argv, &i, arg);
            if (!value || parse_int(value, &options->epochs) != 0) {
                fprintf(stderr, "error: invalid value for %s\n", arg);
                return -1;
            }
        } else if (strcmp(arg, "--batch-size") == 0) {
            value = option_value(argc, argv, &i, arg);
            if (!value || parse_int(value, &options->batch_size) != 0) {
                fprintf(stderr, "error: invalid value for %s\n", arg);
                return -1;
            }
        } else if (strcmp(arg, "--lr") == 0) {
            value = option_value(argc, argv, &i, arg);
            if (!value || parse_float(value, &options->learning_rate) != 0) {
                fprintf(stderr, "error: invalid value for %s\n", arg);
                return -1;
            }
        } else if (strcmp(arg, "--momentum") == 0) {
            value = option_value(argc, argv, &i, arg);
            if (!value || parse_float(value, &options->momentum) != 0) {
                fprintf(stderr, "error: invalid value for %s\n", arg);
                return -1;
            }
        } else if (strcmp(arg, "--seed") == 0) {
            int seed;
            value = option_value(argc, argv, &i, arg);
            if (!value || parse_int(value, &seed) != 0) {
                fprintf(stderr, "error: invalid value for %s\n", arg);
                return -1;
            }
            options->seed = (unsigned int)seed;
        } else if (strcmp(arg, "--data-dir") == 0) {
            value = option_value(argc, argv, &i, arg);
            if (!value) {
                fprintf(stderr, "error: %s requires a value\n", arg);
                return -1;
            }
            options->data_dir = value;
        } else {
            fprintf(stderr, "error: unknown option %s\n", arg);
            return -1;
        }
    }

    if (options->epochs < 0) {
        fprintf(stderr, "error: --epochs must be >= 0\n");
        return -1;
    }
    if (options->batch_size < 1) {
        fprintf(stderr, "error: --batch-size must be >= 1\n");
        return -1;
    }
    if (options->learning_rate <= 0.0f) {
        fprintf(stderr, "error: --lr must be > 0\n");
        return -1;
    }
    if (options->momentum < 0.0f || options->momentum >= 1.0f) {
        fprintf(stderr, "error: --momentum must be in [0, 1)\n");
        return -1;
    }

    return 0;
}

static void normalize_image(const unsigned char *pixels, float *output)
{
    for (int i = 0; i < MNIST_PIXELS; i++)
        output[i] = (float)pixels[i] / 255.0f;
}

static float evaluate(CNN *net, const MNIST *data, int start, int count, float *image)
{
    int correct = 0;

    for (int i = 0; i < count; i++) {
        int index = start + i;
        normalize_image(data->images + (size_t)index * MNIST_PIXELS, image);
        if (cnn_predict(net, image) == data->labels[index])
            correct++;
    }

    return count > 0 ? (float)correct / (float)count * 100.0f : 0.0f;
}

int main(int argc, char **argv)
{
    Options options = {
        .epochs = EPOCHS,
        .batch_size = BATCH_SIZE,
        .learning_rate = LEARNING_RATE,
        .momentum = MOMENTUM,
        .seed = SEED,
        .data_dir = DEFAULT_DATA_DIR,
    };

    if (parse_options(argc, argv, &options) != 0) {
        print_usage(argv[0]);
        return 2;
    }

    if (options.seed == 0)
        options.seed = (unsigned int)time(NULL);

    char images_path[PATH_LENGTH];
    char labels_path[PATH_LENGTH];
    int images_length = snprintf(images_path, sizeof(images_path),
                                 "%s/train-images.idx3-ubyte", options.data_dir);
    int labels_length = snprintf(labels_path, sizeof(labels_path),
                                 "%s/train-labels.idx1-ubyte", options.data_dir);

    if (images_length < 0 || images_length >= (int)sizeof(images_path) ||
        labels_length < 0 || labels_length >= (int)sizeof(labels_path)) {
        fprintf(stderr, "error: --data-dir path is too long\n");
        return 2;
    }

    MNIST data;
    if (mnist_load(images_path, labels_path, &data) != 0)
        return 1;

    srand(options.seed);

    CNN net;
    cnn_init(&net);
    mnist_shuffle(&data, data.count);

    int train_size = (int)(data.count * TRAIN_SPLIT);
    int test_size = data.count - train_size;

    float image[MNIST_PIXELS];
    float accuracy = 0.0f;

    printf("CNN 1x28x28 -> conv3x3/%d -> pool2 -> conv3x3/%d -> pool2 -> %d -> %d\n",
           CNN_CONV1_FILTERS, CNN_CONV2_FILTERS, CNN_FC1_SIZE, CNN_CLASSES);
    printf("Training: %d train / %d test samples, batch %d, lr %.4f, momentum %.2f, "
           "epochs %d, seed %u\n",
           train_size, test_size, options.batch_size, options.learning_rate,
           options.momentum, options.epochs, options.seed);

    for (int epoch = 0; epoch < options.epochs; epoch++) {
        clock_t start = clock();
        double total_loss = 0.0;

        mnist_shuffle(&data, train_size);

        for (int batch_start = 0; batch_start < train_size; batch_start += options.batch_size) {
            int batch_end = batch_start + options.batch_size;
            if (batch_end > train_size)
                batch_end = train_size;

            int batch_count = batch_end - batch_start;
            cnn_zero_grad(&net);

            for (int i = batch_start; i < batch_end; i++) {
                normalize_image(data.images + (size_t)i * MNIST_PIXELS, image);
                cnn_forward(&net, image);
                total_loss += -log(net.probs[data.labels[i]] + 1e-10);
                cnn_backward(&net, data.labels[i]);
            }

            cnn_step(&net, options.learning_rate / (float)batch_count, options.momentum);
        }

        accuracy = evaluate(&net, &data, train_size, test_size, image);

        double seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n",
               epoch + 1, accuracy, (float)(total_loss / train_size), seconds);
    }

    if (options.epochs == 0)
        accuracy = evaluate(&net, &data, train_size, test_size, image);

    printf("Final accuracy: %.2f%%\n", accuracy);

    cnn_free(&net);
    mnist_free(&data);

    return 0;
}
