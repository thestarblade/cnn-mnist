#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

#define IMAGE_MAGIC 0x00000803u
#define LABEL_MAGIC 0x00000801u

static int read_u32_be(FILE *file, unsigned int *value)
{
    unsigned char bytes[4];
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes))
        return -1;

    *value = ((unsigned int)bytes[0] << 24) | ((unsigned int)bytes[1] << 16) |
             ((unsigned int)bytes[2] << 8) | (unsigned int)bytes[3];
    return 0;
}

int mnist_load(const char *images_path, const char *labels_path, MNIST *dataset)
{
    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->count = 0;

    unsigned int magic, count, rows, cols;
    unsigned int label_magic, label_count;
    size_t image_bytes;
    int status = -1;

    FILE *images_file = fopen(images_path, "rb");
    if (!images_file) {
        fprintf(stderr, "mnist: cannot open %s\n", images_path);
        return -1;
    }

    FILE *labels_file = fopen(labels_path, "rb");
    if (!labels_file) {
        fprintf(stderr, "mnist: cannot open %s\n", labels_path);
        fclose(images_file);
        return -1;
    }

    if (read_u32_be(images_file, &magic) != 0 ||
        read_u32_be(images_file, &count) != 0 ||
        read_u32_be(images_file, &rows) != 0 ||
        read_u32_be(images_file, &cols) != 0) {
        fprintf(stderr, "mnist: %s is too short\n", images_path);
        goto done;
    }

    if (magic != IMAGE_MAGIC) {
        fprintf(stderr, "mnist: %s is not an IDX image file\n", images_path);
        goto done;
    }

    if (rows != MNIST_ROWS || cols != MNIST_COLS) {
        fprintf(stderr, "mnist: expected %dx%d images, got %ux%u\n",
                MNIST_ROWS, MNIST_COLS, rows, cols);
        goto done;
    }

    if (count == 0) {
        fprintf(stderr, "mnist: %s contains no images\n", images_path);
        goto done;
    }

    if (read_u32_be(labels_file, &label_magic) != 0 ||
        read_u32_be(labels_file, &label_count) != 0) {
        fprintf(stderr, "mnist: %s is too short\n", labels_path);
        goto done;
    }

    if (label_magic != LABEL_MAGIC) {
        fprintf(stderr, "mnist: %s is not an IDX label file\n", labels_path);
        goto done;
    }

    if (label_count != count) {
        fprintf(stderr, "mnist: %u images but %u labels\n", count, label_count);
        goto done;
    }

    image_bytes = (size_t)count * MNIST_PIXELS;
    dataset->images = malloc(image_bytes);
    dataset->labels = malloc(count);
    if (!dataset->images || !dataset->labels) {
        fprintf(stderr, "mnist: out of memory\n");
        goto done;
    }

    if (fread(dataset->images, 1, image_bytes, images_file) != image_bytes ||
        fread(dataset->labels, 1, count, labels_file) != count) {
        fprintf(stderr, "mnist: truncated IDX data\n");
        goto done;
    }

    dataset->count = (int)count;
    status = 0;

done:
    if (status != 0)
        mnist_free(dataset);
    fclose(images_file);
    fclose(labels_file);
    return status;
}

void mnist_free(MNIST *dataset)
{
    free(dataset->images);
    free(dataset->labels);
    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->count = 0;
}

void mnist_shuffle(MNIST *dataset, int n)
{
    if (n > dataset->count)
        n = dataset->count;

    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        for (int k = 0; k < MNIST_PIXELS; k++) {
            unsigned char pixel = dataset->images[i * MNIST_PIXELS + k];
            dataset->images[i * MNIST_PIXELS + k] = dataset->images[j * MNIST_PIXELS + k];
            dataset->images[j * MNIST_PIXELS + k] = pixel;
        }

        unsigned char label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = label;
    }
}
