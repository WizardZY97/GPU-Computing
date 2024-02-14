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
                int pivotValue = src_image[(j + u) * col + (i + v)]; // Better to access only once to global memory
                gx += pivotValue * sobelX[u + 1][v + 1];
                gy += pivotValue * sobelY[u + 1][v + 1];
            }
        }

        // Calculate the value of gradient
        res_image[j * col + i] = static_cast<int>(sqrtf(gx * gx + gy * gy));
    }
    else
    {   
        res_image[j * col + i] = 0;
    }
}

__global__ void
computeHashKernel(int *hash, int *img, int *res, int n)
{
    __shared__ int sdata[1024];

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
    __syncthreads();
    
    if (tid == 0)
    {
        // Thread 0 writes result for this block to global mem
        res[blockIdx.x] = sdata[0];
    }
}

__global__ void 
computeCosSimKernel(int *hash1, int *hash2, float *dot_product, float *norm_hash1, float *norm_hash2, float num_hashes) {
    __shared__ int sdata1[32];
    __shared__ int sdata2[32];
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if (tid < 32) {
        sdata1[tid] = hash1[idx];   // load data into shared memory
        sdata2[tid] = hash2[idx];   // load data into shared memory
    }

    if (idx < num_hashes) {
        float t_dot_product = static_cast<float>(sdata1[idx]) * static_cast<float>(sdata2[idx]);
        float t_norm_hash1 = static_cast<float>(sdata1[idx]) * static_cast<float>(sdata1[idx]);
        float t_norm_hash2 = static_cast<float>(sdata2[idx]) * static_cast<float>(sdata2[idx]);
        
        // Using atomic operation for the reduction
        atomicAdd(dot_product, t_dot_product);
        atomicAdd(norm_hash1, t_norm_hash1);
        atomicAdd(norm_hash2, t_norm_hash2);
    }
}