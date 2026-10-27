#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "layer.h"
#include "mnist.h"

#define INPUT_SIZE MNIST_PIXELS
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0005f
#define MOMENTUM 0.9f
#define EPOCHS 20
#define TRAIN_SPLIT 0.8

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
    DenseLayer hidden, output;
} Network;

float *train(Network *net, const float *input, int label, float learning_rate) {
    static float final_output[OUTPUT_SIZE];
    float hidden_output[HIDDEN_SIZE];
    float output_grad[OUTPUT_SIZE] = {0};
    float hidden_grad[HIDDEN_SIZE] = {0};

    dense_forward(&net->hidden, input, hidden_output, 1);
    dense_forward(&net->output, hidden_output, final_output, 0);
    softmax_inplace(final_output, OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        output_grad[i] = final_output[i] - (i == label);

    dense_backward(&net->output, hidden_output, output_grad, hidden_grad, learning_rate, MOMENTUM);

    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_grad[i] *= hidden_output[i] > 0 ? 1.0f : 0.0f; /* ReLU derivative */

    dense_backward(&net->hidden, input, hidden_grad, NULL, learning_rate, MOMENTUM);

    return final_output;
}

int predict(const Network *net, const float *input) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

    dense_forward(&net->hidden, input, hidden_output, 1);
    dense_forward(&net->output, hidden_output, final_output, 0);
    softmax_inplace(final_output, OUTPUT_SIZE);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (final_output[i] > final_output[max_index])
            max_index = i;
    }

    return max_index;
}

int main(void) {
    Network net;
    MNIST data;
    float learning_rate = LEARNING_RATE, img[INPUT_SIZE];
    clock_t start, end;
    double cpu_time_used;

    srand(time(NULL));

    dense_init(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    dense_init(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    if (mnist_load(TRAIN_IMG_PATH, TRAIN_LBL_PATH, &data) != 0)
        return 1;

    mnist_shuffle(&data, data.count);

    int train_size = (int)(data.count * TRAIN_SPLIT);
    int test_size = data.count - train_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        start = clock();
        mnist_shuffle(&data, train_size);
        float total_loss = 0;
        for (int i = 0; i < train_size; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;

            float *final_output = train(&net, img, data.labels[i], learning_rate);
            total_loss += -logf(final_output[data.labels[i]] + 1e-10f);
        }
        int correct = 0;
        for (int i = train_size; i < data.count; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
            if (predict(&net, img) == data.labels[i])
                correct++;
        }
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n",
               epoch + 1, (float)correct / test_size * 100, total_loss / train_size, cpu_time_used);
    }

    dense_free(&net.hidden);
    dense_free(&net.output);
    mnist_free(&data);

    return 0;
}
