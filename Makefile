# tinymnist build system
#
#   make          optimized build   -> build/release/main
#   make debug    sanitizer build   -> build/debug/main
#   make run      build and run the release binary
#   make clean    remove build/
#
# Each mode uses its own output directory, so switching between them always
# rebuilds with the right flags instead of reusing stale objects.

CC       ?= cc
CSTD     ?= c11
WARNINGS := -Wall -Wextra
LDLIBS   := -lm

TARGET := main
SRCS   := main.c mnist.c layer.c
OBJS   := $(SRCS:.c=.o)

RELEASE_DIR := build/release
DEBUG_DIR   := build/debug

COMMON_CFLAGS := -std=$(CSTD) $(WARNINGS)

.PHONY: all release debug run clean

all: release

# Keep the flags from the README so accuracy and timing stay comparable.
release: CFLAGS += $(COMMON_CFLAGS) -O3 -march=native -ffast-math -DNDEBUG
release: $(RELEASE_DIR)/$(TARGET)

# Sanitizers catch out-of-bounds indexing, which is the main hazard once
# convolution and pooling start sharing buffers.
debug: CFLAGS += $(COMMON_CFLAGS) -O0 -g3 -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: $(DEBUG_DIR)/$(TARGET)

$(RELEASE_DIR)/$(TARGET): $(addprefix $(RELEASE_DIR)/,$(OBJS))
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(DEBUG_DIR)/$(TARGET): $(addprefix $(DEBUG_DIR)/,$(OBJS))
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(RELEASE_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(DEBUG_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

-include $(wildcard $(RELEASE_DIR)/*.d) $(wildcard $(DEBUG_DIR)/*.d)

run: release
	./$(RELEASE_DIR)/$(TARGET)

clean:
	$(RM) -r build
