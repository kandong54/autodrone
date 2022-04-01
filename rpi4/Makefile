# https://github.com/mbcrawfo/GenericMakefile/blob/master/cpp/Makefile

# The name of the executable to be created
BIN_NAME = main.out
# Compiler used
CXX ?= g++
# Extension of source files used in the project
SRC_EXT = cc
# Path to the source directory
SRC_PATH = src
PROTO_PATH = protos
# Space-separated pkg-config libraries used by this project
LIBS = opencv4 protobuf grpc++ tflite spdlog
# General compiler flags
CXXFLAGS += -Wall -Wextra -O0 -g -std=c++17
# Add additional include paths
INCLUDES = -I $(SRC_PATH)

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

# gRPC code
PROTOS = $(wildcard $(PROTO_PATH)/*.proto)
GRPCPB = $(PROTOS:$(PROTO_PATH)/%.proto=$(SRC_PATH)/%.pb.cc)
GRPCPB := $(GRPCPB) $(GRPCPB:%.pb.cc=%.grpc.pb.cc)

# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
GRPC_OBJECTS = $(GRPCPB:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

# Main rule, checks the executable and symlinks to the output
.PHONY: all
all: dirs proto $(BIN_PATH)/$(BIN_NAME)

# Link the executable
$(BIN_PATH)/$(BIN_NAME): $(OBJECTS) $(GRPC_OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Add dependency files, if they exist
-include $(DEPS)

# Source file rules
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

# Create the directories used in the build
.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(BIN_PATH)

# Generating server code
.PHONY: proto
proto: $(GRPCPB)

$(SRC_PATH)/%.grpc.pb.cc: $(PROTO_PATH)/%.proto
	@echo "Generating: $< -> $@"
	protoc -I $(PROTO_PATH) --grpc_out=$(SRC_PATH) --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` $<

$(SRC_PATH)/%.pb.cc: $(PROTO_PATH)/%.proto
	@echo "Generating: $< -> $@"
	protoc -I $(PROTO_PATH) --cpp_out=$(SRC_PATH) $<

# Removes all build files
.PHONY: clean
clean:
	@echo "Deleting directories"
	rm -rf $(BUILD_PATH) $(BIN_PATH)
	@echo "Deleting generated gRPC code"
	rm -rf $(SRC_PATH)/*.pb.cc $(SRC_PATH)/*.pb.h