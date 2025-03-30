# Compiler & Flags
NVCC = nvcc
CXX = g++
CFLAGS = -O3 -Wall -Wextra -Werror -std=c++17
NVFLAGS = -O3 -lineinfo -Xcompiler "-Wall -Wextra -Werror"

# Paths
SRC_DIR = .
OBJ_DIR = obj
BIN_DIR = bin
KERNELS_DIR = kernels

# Source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(KERNELS_DIR)/*.cu)
HEADERS = $(wildcard $(SRC_DIR)/*.h)

# Object files
C_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(C_SOURCES))
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))
OBJS = $(C_OBJS) $(CU_OBJS)

# Executable
TARGET = $(BIN_DIR)/run

# Build Rules
all: $(TARGET)

# Compiling C files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CFLAGS) -c $< -o $@

# Compiling CUDA files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(KERNELS_DIR)/%.cu $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVFLAGS) -c $< -o $@

# Linking everything
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $^ -o $@

# Clean up object files and binaries
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
