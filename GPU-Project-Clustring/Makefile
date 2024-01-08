CC=g++
NVCC=nvcc

CFLAGS=-Wall -Werror
NVCCFLAGS=-arch=sm_35

BUILD_DIR=.

SRC_DIR=.

CPP_FILES=$(wildcard $(SRC_DIR)/*.cpp)

CPP_OBJ_FILES=clustering.o image_preproc.o LSH.o

TARGET=prog

all: $(TARGET) clean

$(TARGET): $(CPP_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ -I./CImg -lpthread -lX11

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(BUILD_DIR)/*.o
