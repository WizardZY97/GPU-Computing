#include <iostream>
#include <filesystem>
#include <unordered_map>
#include "image_preproc.h"
#include "LSH.h"

int main(int argc, char *argv[])
{
    // Target directory (pictures)
    std::string folder_path = "./Images";

    int num_hashes = 32, dim = SIZE*SIZE;
    LSHCalculator lsh(num_hashes, dim);

    std::unordered_map<std::string, std::vector<int>> mapFileHash;
    std::vector<std::string> files;

    for (const auto &entry : std::filesystem::directory_iterator(folder_path))
    {
        // Check if the file is regular
        if (std::filesystem::is_regular_file(entry))
        {
            std::string s = entry.path().string();
            const char *filename = s.c_str();
            
            std::vector<std::vector<int>> input_image = readImageToVec(filename);
            std::vector<std::vector<int>> feature_image = applySobel(input_image);
            std::vector<int> feature_vec = flatten(feature_image);

            std::vector<int> hash = lsh.computeHash(feature_vec);

            std::pair<std::string, std::vector<int>> one_pair(s, hash);

            files.push_back(s);
            mapFileHash.insert(one_pair);
        }
    }

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

    return 0;
}