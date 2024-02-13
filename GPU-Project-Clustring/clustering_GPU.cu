#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <cuda_runtime.h>

#include "image_preproc.h"
#include "kernel_func.cuh"

int main(int argc, char *argv[])
{
    // Utilities
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Target directory (pictures)
    std::string folder_path = "./Images";

    int num_hashes = 32, dim = SIZE*SIZE;

    /******************* Build Hash Functions Start *******************/

    // Too complex to get the correct random number in the Kernel function
    // Abandon the parallel way to create the Hash functions

    int **hash_functions = new int*[num_hashes];

    for (int i = 0; i < num_hashes; i++)
    {
        hash_functions[i] = new int[dim];

        srand(time(nullptr) + i);
        for (int j = 0; j < dim; j++)
        {
            hash_functions[i][j] = (rand() % 2 == 0 ? 1 : -1);
        }
    }
    
    /******************* Build Hash Functions End *********************/

    std::unordered_map<std::string, std::vector<int>> mapFileHash;
    std::vector<std::string> files;

    for (const auto &entry : std::filesystem::directory_iterator(folder_path))
    {
        // Check if the file is regular
        if (std::filesystem::is_regular_file(entry))
        {
            std::string s = entry.path().string();
            const char *filename = s.c_str();

            /******************* Kernel Soble Feature Start *******************/

            int *original_image_arr = new int[dim];
            int *feature_image_arr = new int[dim];

            readImageToArr(filename, original_image_arr);

            int *d_original_image_arr, *d_feature_image_arr; // device copies

            // allocate the GPU memory space
            cudaMalloc((void **)&d_original_image_arr, sizeof(int)*dim); 
            cudaMalloc((void **)&d_feature_image_arr, sizeof(int)*dim); 

            // copy source data from CPU memory to GPU memory
            cudaMemcpy(d_original_image_arr, original_image_arr, sizeof(int)*dim, cudaMemcpyHostToDevice);

            // Define the size of the Grid and the Block
            dim3 dimBlockSoble(32, 32, 1);
            dim3 dimGridSoble(ceil(SIZE / 32.0), ceil(SIZE / 32.0), 1);

            // Execute the kernel fucntion of Soble
            float msecSoble = 0.0;
            cudaEventRecord(start);
            applySobelKernel<<<dimGridSoble, dimBlockSoble>>>(d_original_image_arr, d_feature_image_arr, SIZE, SIZE);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&msecSoble, start, stop);

            // copy result data from GPU memory back to CPU memory
            cudaMemcpy(feature_image_arr, d_feature_image_arr, sizeof(int)*dim, cudaMemcpyDeviceToHost);

            // Free the allocated GPU memory
            cudaFree(d_original_image_arr);
            cudaFree(d_feature_image_arr);

            /******************* Kernel Soble Feature End *********************/

            /******************* Kernel Compute Hash Start *******************/

            std::vector<int> hashes_image;

            int *d_hash_function, *d_hash_value_collector; // device copies

            // Define the size of the Grid and the Block
            dim3 dimBlockHashComp(1024, 1, 1);
            dim3 dimGridHashComp(ceil(dim / 1024.0), 1, 1);
            
            float msecHash = 0.0;

            for (int i = 0; i < num_hashes; i++) 
            {
                int *hash_value_collector = new int[SIZE];// Equal to the number of blocks
                
                // allocate the GPU memory space
                cudaMalloc((void **)&d_hash_function, sizeof(int)*dim); 
                cudaMalloc((void **)&d_feature_image_arr, sizeof(int)*dim); 
                cudaMalloc((void **)&d_hash_value_collector, sizeof(int)*SIZE);

                // copy source data from CPU memory to GPU memory
                cudaMemcpy(d_hash_function, hash_functions[i], sizeof(int)*dim, cudaMemcpyHostToDevice);
                cudaMemcpy(d_feature_image_arr, feature_image_arr, sizeof(int)*dim, cudaMemcpyHostToDevice);

                // Execute the kernel fucntion of Hash
                float temp = 0.0;
                cudaEventRecord(start);
                computeHashKernel<<<dimGridHashComp, dimBlockHashComp>>>(d_hash_function, d_feature_image_arr, d_hash_value_collector, dim);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&temp, start, stop);
                msecHash += temp;

                // copy result data from GPU memory back to CPU memory
                cudaMemcpy(hash_value_collector, d_hash_value_collector, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);

                int hash_value = 0;
                for (int j = 0; j < SIZE; j++)
                {
                    // Accumulation of each block
                    hash_value += hash_value_collector[j];
                }
                hashes_image.push_back(hash_value);

                // Free the allocated GPU memory
                cudaFree(d_hash_function);
                cudaFree(d_hash_value_collector);
                cudaFree(d_feature_image_arr);

                delete hash_value_collector;
            }

            /******************* Kernel Compute Hash End *********************/

            std::pair<std::string, std::vector<int>> one_pair(s, hashes_image);

            files.push_back(s);
            mapFileHash.insert(one_pair);

            delete original_image_arr;
            delete feature_image_arr;

            std::cout << "Soble Kernel execution time of " << s << ": " << msecSoble << " milliseconds\n";
            std::cout << "Hash Kernel execution time of " << s << ": " << msecHash << " milliseconds\n";
        }
    }

    for (int i = 0; i < num_hashes; i++)
    {
        delete hash_functions[i];
    }
    delete hash_functions;

    float msecCosSim = 0.0;
    for (long unsigned int i = 0; i < files.size(); i++)
    {
        for (long unsigned int j = i + 1; j < files.size(); j++)
        {
            std::vector<int> hash1 = mapFileHash.at(files[i]);
            std::vector<int> hash2 = mapFileHash.at(files[j]);

            float dot_product = 0, norm_hash1 = 0, norm_hash2 = 0;

            int *d_hash1 = nullptr, *d_hash2 = nullptr; // device copies for the input
            float *d_dot_product = nullptr, *d_norm_hash1 = nullptr, *d_norm_hash2 = nullptr;    // device copies for the output

            // allocate the GPU memory space
            cudaMalloc((void **)&d_hash1, sizeof(int)*num_hashes);
            cudaMalloc((void **)&d_hash2, sizeof(int)*num_hashes);
            cudaMalloc((void **)&d_dot_product, sizeof(float));
            cudaMalloc((void **)&d_norm_hash1, sizeof(float));
            cudaMalloc((void **)&d_norm_hash2, sizeof(float));

            // copy source data from CPU memory to GPU memory
            cudaMemcpy(d_hash1, hash1.data(), sizeof(int)*num_hashes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_hash2, hash2.data(), sizeof(int)*num_hashes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dot_product, &dot_product, sizeof(float), cudaMemcpyHostToDevice);// To avoid the random initial value 
            cudaMemcpy(d_norm_hash1, &norm_hash1, sizeof(float), cudaMemcpyHostToDevice);  // To avoid the random initial value 
            cudaMemcpy(d_norm_hash2, &norm_hash2, sizeof(float), cudaMemcpyHostToDevice);  // To avoid the random initial value 

            // Define the size of the Grid and the Block (perfectly matching)
            dim3 dimBlockSim(32, 1, 1);
            dim3 dimGridSim(ceil(num_hashes / 32.0), 1, 1);

            float temp = 0.0;
            cudaEventRecord(start);
            computeCosSimKernel<<<dimGridSim, dimBlockSim>>>(d_hash1, d_hash2, d_dot_product, d_norm_hash1, d_norm_hash2, num_hashes);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&temp, start, stop);
            msecCosSim += temp;

            // copy result data from GPU memory back to CPU memory
            cudaMemcpy(&dot_product, d_dot_product, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&norm_hash1, d_norm_hash1, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&norm_hash2, d_norm_hash2, sizeof(float), cudaMemcpyDeviceToHost);

            // Free the allocated GPU memory
            cudaFree(d_hash1);
            cudaFree(d_hash2);
            cudaFree(d_dot_product);
            cudaFree(d_norm_hash1);
            cudaFree(d_norm_hash2);

            // Cosine similarity coefficient
            double sim = dot_product / (sqrt(norm_hash1) * sqrt(norm_hash2));

            std::cout << "Similarity between " << files[i] << " and " << files[j] << " : " << sim << std::endl;
        }
    }
    std::cout << "Total Cosine Similarity Kernel execution time: " << msecCosSim << " milliseconds\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}