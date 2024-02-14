#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <chrono>

#include "image_preproc.h"
#include "LSH.h"

int main(int argc, char *argv[])
{
    // Target directory (pictures)
    std::string folder_path = "./Images";

    int num_hashes = 32, dim = SIZE*SIZE;   // Define the number of hash functions & the size of one hash function
    LSHCalculator lsh(num_hashes, dim);     // Initialize the hash functions

    std::unordered_map<std::string, std::vector<int>> mapFileHash;
    std::vector<std::string> files;

    // Iterate all the images in the folder
    for (const auto &entry : std::filesystem::directory_iterator(folder_path))
    {
        // Check if the file is regular
        if (std::filesystem::is_regular_file(entry))
        {
            std::string s = entry.path().string();
            const char *filename = s.c_str();
            
            std::vector<std::vector<int>> input_image = readImageToVec(filename);

            auto start_Soble = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<int>> feature_image = applySobel(input_image);
            auto stop_Soble = std::chrono::high_resolution_clock::now();
            auto duration_Soble = std::chrono::duration_cast<std::chrono::milliseconds>(stop_Soble - start_Soble);

            std::vector<int> feature_vec = flatten(feature_image);

            auto start_Hash = std::chrono::high_resolution_clock::now();
            std::vector<int> hash = lsh.computeHash(feature_vec);
            auto stop_Hash = std::chrono::high_resolution_clock::now();
            auto duration_Hash = std::chrono::duration_cast<std::chrono::milliseconds>(stop_Hash - start_Hash);

            std::pair<std::string, std::vector<int>> one_pair(s, hash);

            files.push_back(s);
            mapFileHash.insert(one_pair);

            std::cout << "Function Soble execution time of " << s << ": " << duration_Soble.count() << " milliseconds\n";
            std::cout << "Function Hash execution time of " << s << ": " << duration_Hash.count() << " milliseconds\n";
        }
    }

    auto start_CosSim = std::chrono::high_resolution_clock::now();
    // Compute the similarity for each pair of images
    for (long unsigned int i = 0; i < files.size(); i++)
    {
        for (long unsigned int j = i + 1; j < files.size(); j++)
        {
            std::vector<int> hash1 = mapFileHash.at(files[i]);
            std::vector<int> hash2 = mapFileHash.at(files[j]);

            double similarity = lsh.calculateSimilarity(hash1, hash2);

            std::cout << "Similarity between " << files[i] << " and " << files[j] << " : " << similarity << std::endl;
        }
    }
    auto stop_CosSim = std::chrono::high_resolution_clock::now();
    auto duration_CosSim = std::chrono::duration_cast<std::chrono::milliseconds>(stop_CosSim - start_CosSim);
    std::cout << "Total Function Cosine Similarity execution time: " << duration_CosSim.count() << " milliseconds\n";

    return 0;
}