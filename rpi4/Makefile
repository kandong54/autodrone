# https://github.com/mbcrawfo/GenericMakefile/blob/master/cpp/Makefile

# The name of the executable to be created
BIN_NAME = main.out
# Compiler used
CXX ?= g++
# Extension of source files used in the project
SRC_EXT = cc
# General compiler flags
CXXFLAGS += -Wall -Wextra -O0 -g -std=c++17
# Path to the source directory
SRC_PATH = src
# Space-separated pkg-config libraries used by this project
LIBS = opencv4

# Build and output paths
BUILD_PATH = build
BIN_PATH = bin

# Append pkg-config specific libraries if need be
ifneq ($(LIBS),)
	CXXFLAGS += $(shell pkg-config --cflags $(LIBS))
	LDFLAGS += $(shell pkg-config --libs $(LIBS))
endif


# Find all source files in the source directory, sorted by most
# recently modified
SOURCES = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' -printf '%T@\t%p\n' \
					| sort -k 1nr | cut -f2-)


# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

# Main rule, checks the executable and symlinks to the output
.PHONY: all
all: dirs $(BIN_PATH)/$(BIN_NAME)

# Link the executable
$(BIN_PATH)/$(BIN_NAME): $(OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Source file rules
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create the directories used in the build
.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(BIN_PATH)

# Removes all build files
.PHONY: clean
clean: 
	@echo "Deleting directories"
	rm -rf $(BUILD_PATH) $(BIN_PATH)