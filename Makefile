# tinymnist build system
#
#   make          optimized build   -> build/release/main
#   make debug    sanitizer build   -> build/debug/main
#   make test     build and run the unit tests
#   make run      build and run the release binary
#   make clean    remove build/
#
# Each mode uses its own output directory, so switching between them always
# rebuilds with the right flags instead of reusing stale objects.

CC       ?= cc
CSTD     ?= c11
WARNINGS := -Wall -Wextra
LDLIBS   := -lm

TARGET   := main
LIB_SRCS := mnist.c layer.c conv.c cnn.c
APP_SRCS := main.c $(LIB_SRCS)

RELEASE_DIR := build/release
DEBUG_DIR   := build/debug
TEST_DIR    := build/test
TEST_BIN_DIR := $(TEST_DIR)/bin

TEST_SRCS := $(wildcard tests/*.c)
TEST_BINS := $(patsubst tests/%.c,$(TEST_BIN_DIR)/%,$(TEST_SRCS))

RELEASE_OBJS := $(addprefix $(RELEASE_DIR)/,$(APP_SRCS:.c=.o))
DEBUG_OBJS   := $(addprefix $(DEBUG_DIR)/,$(APP_SRCS:.c=.o))
TEST_OBJS    := $(addprefix $(TEST_DIR)/,$(LIB_SRCS:.c=.o))

COMMON_CFLAGS := -std=$(CSTD) $(WARNINGS)

.PHONY: all release debug test run clean

all: release

# Keep the flags from the README so accuracy and timing stay comparable.
release: CFLAGS += $(COMMON_CFLAGS) -O3 -march=native -ffast-math -DNDEBUG
release: $(RELEASE_DIR)/$(TARGET)

# Sanitizers catch out-of-bounds indexing, which is the main hazard once
# convolution and pooling start sharing buffers.
debug: CFLAGS += $(COMMON_CFLAGS) -O0 -g3 -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: $(DEBUG_DIR)/$(TARGET)

test: CFLAGS += $(COMMON_CFLAGS) -O1 -g3 -fsanitize=address,undefined
test: LDFLAGS += -fsanitize=address,undefined
test: $(TEST_BINS)
	@for binary in $(TEST_BINS); do \
		echo "== $$binary"; \
		./$$binary || exit 1; \
	done

$(RELEASE_DIR)/$(TARGET): $(RELEASE_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(DEBUG_DIR)/$(TARGET): $(DEBUG_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(RELEASE_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(DEBUG_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(TEST_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

$(TEST_BIN_DIR)/%: tests/%.c $(TEST_OBJS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(TEST_OBJS) $(LDLIBS)

-include $(wildcard $(RELEASE_DIR)/*.d) $(wildcard $(DEBUG_DIR)/*.d) $(wildcard $(TEST_DIR)/*.d)

run: release
	./$(RELEASE_DIR)/$(TARGET)

clean:
	$(RM) -r build
