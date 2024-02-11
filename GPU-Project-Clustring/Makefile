CC=g++
NVCC=nvcc

CFLAGS=-Wall -Werror
NVCCFLAGS = -g -G -Xcompiler -Wall

BUILD_DIR=.

SRC_DIR=.

CPP_FILES=$(wildcard $(SRC_DIR)/*.cpp)

CPP_OBJ_FILES=clustering_CPU.o image_preproc.o LSH.o

TARGET=prog

TARGET_GPU=prog_gpu

SRC=clustering_GPU.cu image_preproc.o

all: $(TARGET) clean

$(TARGET): $(CPP_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ -I./CImg -lpthread -lX11

$(TARGET_GPU): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET_GPU) -I./CImg -lpthread -lX11

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(BUILD_DIR)/*.o
