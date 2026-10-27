#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "cnn.h"

/* Binary checkpoint of every weight and bias, prefixed with a magic string and
 * the layer dimensions so a mismatched file is rejected instead of silently
 * corrupting the network.
 *
 * The floats are written in host byte order; checkpoints are meant to be moved
 * between runs on the same machine, not as a portable interchange format.
 */
int checkpoint_save(const CNN *net, const char *path);
int checkpoint_load(CNN *net, const char *path);

#endif
