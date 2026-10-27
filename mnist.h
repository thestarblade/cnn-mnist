#ifndef MNIST_H
#define MNIST_H

#define MNIST_ROWS 28
#define MNIST_COLS 28
#define MNIST_PIXELS (MNIST_ROWS * MNIST_COLS)

typedef struct {
    unsigned char *images; /* count * MNIST_PIXELS, row-major */
    unsigned char *labels; /* count */
    int count;
} MNIST;

/* Reads a pair of IDX files. Returns 0 on success and -1 on failure, in which
   case the reason is printed to stderr and the dataset stays empty. */
int mnist_load(const char *images_path, const char *labels_path, MNIST *dataset);

void mnist_free(MNIST *dataset);

/* Fisher-Yates shuffle over the first n samples. */
void mnist_shuffle(MNIST *dataset, int n);

#endif
