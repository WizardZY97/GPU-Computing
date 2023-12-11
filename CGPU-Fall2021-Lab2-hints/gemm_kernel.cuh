// Set tile width for static declaration of shared memory matrices
#define TILE_WIDTH 32

__global__ void gemm_naive(float *C, float *A, float *B, int wA, int wB) {
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

  for (int k = 0; k < wA; k++) {
    accu += A[i * wA + k] * B[k * wB + j];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[i * wB + j] = accu;
}

__global__ void gemm_tiling(float *C, float *A, float *B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * TILE_WIDTH * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = TILE_WIDTH;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = TILE_WIDTH * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = TILE_WIDTH * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all tiles of A and B required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory to shared memory;
    // each thread loads one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[tx][ty] = B[b + wB * tx + ty];

    // Synchronize threads to insure that all warps have loaded the data
    __syncthreads();

    // Multiply the two matrices together:
    // each thread computes one element of the block sub-matrix
    for (int k = 0; k < TILE_WIDTH; ++k)
      Csub += As[ty][k] * Bs[k][tx];

    // Synchronize to insure that the preceding computation is done
    // before loading two new tiles of A and B for the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * TILE_WIDTH * by + TILE_WIDTH * bx;
  C[c + wB * ty + tx] = Csub;
}

__global__ void gemm_coalescing(float *C, float *A, float *B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Declaration of the shared memory array As used to
  // store the tile of A
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];

  // Declaration of the shared memory array Bs used to
  // store the tilex of B
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  // Index of the first tile of A processed by the block
  int aBegin = wA * TILE_WIDTH * by;

  // Index of the last tile of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the tiles of A
  int aStep = TILE_WIDTH;

  // Index of the first tile of B processed by the block
  int bBegin = TILE_WIDTH * bx;

  // Step size used to iterate through the tiles of B
  int bStep = TILE_WIDTH * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the tiles of A and B required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory to shared memory;
    // each thread loads one element of each matrix
    /// @note Add coalesced access for tile Bs - 
    /// A and B are loaded in the same order in shared memory
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[tx][ty] = B[b + wB * ty + tx];

    // Synchronize threads to insure that all warps have loaded the data
    __syncthreads();

    // Multiply the two matrices together:
    // each thread computes one element of the block sub-matrix
    for (int k = 0; k < TILE_WIDTH; ++k)
      Csub += As[ty][k] * Bs[tx][k];

    // Synchronize to insure that the preceding computation is done
    // before loading two new tiles of A and B for the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * TILE_WIDTH * by + TILE_WIDTH * bx;
  C[c + wB * ty + tx] = Csub;
}

__global__ void gemm_nobankconlict( float* C, float* A, float* B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Declaration of the shared memory array As used to
  // store the tile of A
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];

  // Declaration of the shared memory array Bs used to
  // store the tilex of B
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  // Index of the first tile of A processed by the block
  int aBegin = wA * TILE_WIDTH * by;

  // Index of the last tile of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the tiles of A
  int aStep = TILE_WIDTH;

  // Index of the first tile of B processed by the block
  int bBegin = TILE_WIDTH * bx;

  // Step size used to iterate through the tiles of B
  int bStep = TILE_WIDTH * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the tiles of A and B required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Load the matrices from device memory to shared memory;
    // each thread loads one element of each matrix
    /// @note Modify writing in shared memory. As and Bs follow the same order.
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together:
    // each thread computes one element of the block sub-matrix
    /// @note Remove Shared Mem Bank conflict by adjusting to the new order of coefficient in shared memory for tile Bs
    for (int k = 0; k < TILE_WIDTH; ++k)
      Csub += As[ty][k] * Bs[k][tx];

    // Synchronize to insure that the preceding computation is done
    // before loading two new tiles of A and B for the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * TILE_WIDTH * by + TILE_WIDTH * bx;
  C[c + wB * ty + tx] = Csub;
}
