#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

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

int main(void)
{
    MNIST data;
    CNN net;
    float image[MNIST_PIXELS];
    double total_loss;
    float accuracy = 0.0f;
    clock_t start, end;
    double seconds;
    unsigned int seed = SEED != 0 ? (unsigned int)SEED : (unsigned int)time(NULL);

    srand(seed);

    if (mnist_load(TRAIN_IMG_PATH, TRAIN_LBL_PATH, &data) != 0)
        return 1;

    cnn_init(&net);
    mnist_shuffle(&data, data.count);

    int train_size = (int)(data.count * TRAIN_SPLIT);
    int test_size = data.count - train_size;

    printf("CNN 1x28x28 -> conv3x3/%d -> pool2 -> conv3x3/%d -> pool2 -> %d -> %d\n",
           CNN_CONV1_FILTERS, CNN_CONV2_FILTERS, CNN_FC1_SIZE, CNN_CLASSES);
    printf("Training: %d train / %d test samples, batch %d, lr %.4f, momentum %.2f, "
           "epochs %d, seed %u\n",
           train_size, test_size, BATCH_SIZE, LEARNING_RATE, MOMENTUM, EPOCHS, seed);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        start = clock();
        mnist_shuffle(&data, train_size);
        total_loss = 0.0;

        for (int batch_start = 0; batch_start < train_size; batch_start += BATCH_SIZE) {
            int batch_end = batch_start + BATCH_SIZE;
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

            cnn_step(&net, LEARNING_RATE / (float)batch_count, MOMENTUM);
        }

        accuracy = evaluate(&net, &data, train_size, test_size, image);

        end = clock();
        seconds = (double)(end - start) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n",
               epoch + 1, accuracy, (float)(total_loss / train_size), seconds);
    }

    printf("Final accuracy: %.2f%%\n", accuracy);

    cnn_free(&net);
    mnist_free(&data);

    return 0;
}
