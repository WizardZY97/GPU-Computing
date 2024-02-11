__global__ void
applySobelKernel(int *src_image, int *res_image, int row, int col)
{
    // Define the Sobel kernels (calculators) on the axe X, Y
    int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && j >= 1 && i < row - 1 && j < col - 1)
    {
        int gx = 0, gy = 0;

        // Calculate the gradient of pixel(i, j) on the axe X, Y
        for (int u = -1; u <= 1; ++u)
        {
            for (int v = -1; v <= 1; ++v)
            {
                int pivotValue = src_image[(i + u * row) + (j + v)]; // Better to access only once to global memory
                gx += pivotValue * sobelX[u + 1][v + 1];
                gy += pivotValue * sobelY[u + 1][v + 1];
            }
        }

        // Calculate the value of gradient
        res_image[i * row + j] = static_cast<int>(sqrt(gx * gx + gy * gy));
    }
}

__global__ void
computeHashKernel(int *hash, int *img, int *res, int n)
{
    __shared__ int sdata[blockDim.x];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n)
    {
        // do the hash computation 
        // put the results into the shared memory
        sdata[tid] = hash[idx] * img[idx];
    }
	__syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            // Sequential Addressing Reduction 
            sdata[tid] += sdata[tid + stride];
        }
	    __syncthreads();
    }
    
    if (tid == 0)
    {
        // Thread 0 writes result for this block to global mem
        res[blockDim.x] = sdata[0];
    }
}