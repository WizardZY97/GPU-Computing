#include "matrix_utils.h"

__global__ void
gemm_naive( float* C, float* A, float* B, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulate row i of A and column j of B
  int i = by * blockDim.y + ty;
  int j = bx * blockDim.x + tx;

  float accu = 0.0;

  for(int k=0; k<wA; k++){
    accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ i * wB + j ] = accu;

}

__global__ void
gemm_shared( float* C, float* A, float* B, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulate row i of A and column j of B
  int i = by * blockDim.y + ty; // row
  int j = bx * blockDim.x + tx; // col

  // Define the size of a tile
  const int TILE_SIZE = BLOCK_SIZE;
  __shared__ int tileA[TILE_SIZE][TILE_SIZE];
  __shared__ int tileB[TILE_SIZE][TILE_SIZE];

  // Define the number of tiles
  int tiles = wA / TILE_SIZE;

  float accu = 0.0;

  for (int l = 0; l < tiles; l++)
  {
    // load data from global memory to shared memory
    tileA[threadIdx.x][threadIdx.y] = A[(i * wA) + (l * blockDim.x + threadIdx.x)];
    tileB[threadIdx.x][threadIdx.y] = B[((l * blockDim.y + threadIdx.y) * wB) + j];

    // sync to wait for all threads in one block to finish loading datas
    __syncthreads();

    // sub-matrix multiply
    for (int k = 0; k < TILE_SIZE; k++) 
    {
      accu = accu + tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    // sync to wait for all threads in one block to finish compute
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ i * wB + j ] = accu;

}