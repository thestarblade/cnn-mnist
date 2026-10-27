#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECKPOINT_MAGIC "CNN1"
#define CHECKPOINT_VERSION 1u
#define HEADER_FIELDS 8 /* version + four sizes + kernel + height + width */
#define HEADER_SIZE (4 + HEADER_FIELDS * 4)

static size_t conv_weight_count(const ConvLayer *layer)
{
    return (size_t)layer->out_c * layer->in_c * layer->k * layer->k;
}

static size_t dense_weight_count(const DenseLayer *layer)
{
    return (size_t)layer->input_size * layer->output_size;
}

static size_t parameter_count(const CNN *net)
{
    return conv_weight_count(&net->conv1) + net->conv1.out_c +
           conv_weight_count(&net->conv2) + net->conv2.out_c +
           dense_weight_count(&net->fc1) + net->fc1.output_size +
           dense_weight_count(&net->fc2) + net->fc2.output_size;
}

static int write_u32(FILE *file, unsigned int value)
{
    return fwrite(&value, sizeof(value), 1, file) == 1 ? 0 : -1;
}

static int read_u32(FILE *file, unsigned int *value)
{
    return fread(value, sizeof(*value), 1, file) == 1 ? 0 : -1;
}

static int write_floats(FILE *file, const float *data, size_t count)
{
    return fwrite(data, sizeof(float), count, file) == count ? 0 : -1;
}

static int read_floats(FILE *file, float *data, size_t count)
{
    return fread(data, sizeof(float), count, file) == count ? 0 : -1;
}

int checkpoint_save(const CNN *net, const char *path)
{
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "checkpoint: cannot open %s for writing\n", path);
        return -1;
    }

    int status = 0;

    if (fwrite(CHECKPOINT_MAGIC, 1, 4, file) != 4)
        status = -1;

    if (status == 0 && (write_u32(file, CHECKPOINT_VERSION) != 0 ||
                        write_u32(file, (unsigned int)net->conv1.out_c) != 0 ||
                        write_u32(file, (unsigned int)net->conv2.out_c) != 0 ||
                        write_u32(file, (unsigned int)net->fc1.output_size) != 0 ||
                        write_u32(file, (unsigned int)net->fc2.output_size) != 0 ||
                        write_u32(file, (unsigned int)net->conv1.k) != 0 ||
                        write_u32(file, (unsigned int)net->conv1.in_h) != 0 ||
                        write_u32(file, (unsigned int)net->conv1.in_w) != 0))
        status = -1;

    if (status == 0 &&
        (write_floats(file, net->conv1.weights, conv_weight_count(&net->conv1)) != 0 ||
         write_floats(file, net->conv1.biases, (size_t)net->conv1.out_c) != 0 ||
         write_floats(file, net->conv2.weights, conv_weight_count(&net->conv2)) != 0 ||
         write_floats(file, net->conv2.biases, (size_t)net->conv2.out_c) != 0 ||
         write_floats(file, net->fc1.weights, dense_weight_count(&net->fc1)) != 0 ||
         write_floats(file, net->fc1.biases, (size_t)net->fc1.output_size) != 0 ||
         write_floats(file, net->fc2.weights, dense_weight_count(&net->fc2)) != 0 ||
         write_floats(file, net->fc2.biases, (size_t)net->fc2.output_size) != 0))
        status = -1;

    if (fclose(file) != 0)
        status = -1;

    if (status != 0)
        fprintf(stderr, "checkpoint: failed to write %s\n", path);

    return status;
}

int checkpoint_load(CNN *net, const char *path)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "checkpoint: cannot open %s for reading\n", path);
        return -1;
    }

    long expected_size = (long)HEADER_SIZE + (long)(parameter_count(net) * sizeof(float));
    if (fseek(file, 0, SEEK_END) != 0 || ftell(file) != expected_size) {
        fprintf(stderr, "checkpoint: %s does not match this network\n", path);
        fclose(file);
        return -1;
    }
    rewind(file);

    char magic[4];
    unsigned int version, conv1_filters, conv2_filters, fc1_size, classes;
    unsigned int kernel, in_h, in_w;
    int status = 0;

    if (fread(magic, 1, 4, file) != 4 ||
        read_u32(file, &version) != 0 ||
        read_u32(file, &conv1_filters) != 0 ||
        read_u32(file, &conv2_filters) != 0 ||
        read_u32(file, &fc1_size) != 0 ||
        read_u32(file, &classes) != 0 ||
        read_u32(file, &kernel) != 0 ||
        read_u32(file, &in_h) != 0 ||
        read_u32(file, &in_w) != 0) {
        status = -1;
    }

    if (status == 0 &&
        (magic[0] != CHECKPOINT_MAGIC[0] || magic[1] != CHECKPOINT_MAGIC[1] ||
         magic[2] != CHECKPOINT_MAGIC[2] || magic[3] != CHECKPOINT_MAGIC[3] ||
         version != CHECKPOINT_VERSION ||
         conv1_filters != (unsigned int)net->conv1.out_c ||
         conv2_filters != (unsigned int)net->conv2.out_c ||
         fc1_size != (unsigned int)net->fc1.output_size ||
         classes != (unsigned int)net->fc2.output_size ||
         kernel != (unsigned int)net->conv1.k ||
         in_h != (unsigned int)net->conv1.in_h ||
         in_w != (unsigned int)net->conv1.in_w))
        status = -1;

    if (status == 0 &&
        (read_floats(file, net->conv1.weights, conv_weight_count(&net->conv1)) != 0 ||
         read_floats(file, net->conv1.biases, (size_t)net->conv1.out_c) != 0 ||
         read_floats(file, net->conv2.weights, conv_weight_count(&net->conv2)) != 0 ||
         read_floats(file, net->conv2.biases, (size_t)net->conv2.out_c) != 0 ||
         read_floats(file, net->fc1.weights, dense_weight_count(&net->fc1)) != 0 ||
         read_floats(file, net->fc1.biases, (size_t)net->fc1.output_size) != 0 ||
         read_floats(file, net->fc2.weights, dense_weight_count(&net->fc2)) != 0 ||
         read_floats(file, net->fc2.biases, (size_t)net->fc2.output_size) != 0))
        status = -1;

    fclose(file);

    if (status != 0)
        fprintf(stderr, "checkpoint: %s is not a valid checkpoint\n", path);

    return status;
}
