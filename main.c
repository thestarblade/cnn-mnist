#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
    float *weights, *biases, *weight_momentum, *bias_momentum;
    int input_size, output_size;
} Layer;

typedef struct {
    Layer hidden, output;
} Network;


void softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

void init_layer(Layer *layer, int in_size, int out_size) {
    int n = in_size * out_size;
    float scale = sqrtf(2.0f / in_size);

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float));
    layer->weight_momentum = calloc(n, sizeof(float));
    layer->bias_momentum = calloc(out_size, sizeof(float));

    for (int i = 0; i < n; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

void forward(Layer *layer, float *input, float *output, int relu) {
    for (int i = 0; i < layer->output_size; i++)
        output[i] = layer->biases[i];

    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        if (in_j == 0.0f) continue;
        float *weight_row = &layer->weights[j * layer->output_size];
        for (int i = 0; i < layer->output_size; i++) {
            output[i] += in_j * weight_row[i];
        }
    }

    if (relu)
        for (int i = 0; i < layer->output_size; i++)
            output[i] = output[i] > 0 ? output[i] : 0;
}


void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
    for (int j = 0; j < layer->input_size; j++) {
        float in_j = input[j];
        if (in_j == 0.0f && !input_grad) continue;
        float *weight_row = &layer->weights[j * layer->output_size];
        float *momentum_row = &layer->weight_momentum[j * layer->output_size];
        float grad_j = 0.0f;
        for (int i = 0; i < layer->output_size; i++) {
            grad_j += output_grad[i] * weight_row[i];
            momentum_row[i] = MOMENTUM * momentum_row[i] + lr * output_grad[i] * in_j;
            weight_row[i] -= momentum_row[i];
        }
        if (input_grad)
            input_grad[j] = grad_j;
    }

    for (int i = 0; i < layer->output_size; i++) {
        layer->bias_momentum[i] = MOMENTUM * layer->bias_momentum[i] + lr * output_grad[i];
        layer->biases[i] -= layer->bias_momentum[i];
    }
}


float* train(Network *net, float *input, int label, float lr) {
    static float final_output[OUTPUT_SIZE];
    float hidden_output[HIDDEN_SIZE];
    float output_grad[OUTPUT_SIZE] = {0}, hidden_grad[HIDDEN_SIZE] = {0};

    forward(&net->hidden, input, hidden_output, 1);
    forward(&net->output, hidden_output, final_output, 0);
    softmax(final_output, OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        output_grad[i] = final_output[i] - (i == label);

    backward(&net->output, hidden_output, output_grad, hidden_grad, lr);

    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0;  // ReLU derivative

    backward(&net->hidden, input, hidden_grad, NULL, lr);

    return final_output;
}

int predict(Network *net, float *input) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

    forward(&net->hidden, input, hidden_output, 1);
    forward(&net->output, hidden_output, final_output, 0);
    softmax(final_output, OUTPUT_SIZE);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (final_output[i] > final_output[max_index])
            max_index = i;

    return max_index;
}

int main() {
    Network net;
    MNIST data;
    float learning_rate = LEARNING_RATE, img[INPUT_SIZE];
    clock_t start, end;
    double cpu_time_used;

    srand(time(NULL));

    init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

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

            float* final_output = train(&net, img, data.labels[i], learning_rate);
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
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", 
               epoch + 1, (float)correct / test_size * 100, total_loss / train_size, cpu_time_used);
    }

    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.hidden.weight_momentum);
    free(net.hidden.bias_momentum);
    free(net.output.weights);
    free(net.output.biases);
    free(net.output.weight_momentum);
    free(net.output.bias_momentum);
    mnist_free(&data);

    return 0;
}