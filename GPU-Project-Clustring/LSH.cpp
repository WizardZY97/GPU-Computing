#include "LSH.h"

/******************* LSHHashFunction ***********************/

LSHHashFunction::LSHHashFunction(int dim)
{
    // Set the dimension of the hash function
    m_dim = dim;

    // Initialize a hash funtion
    srand(time(nullptr));
    for (int i = 0; i < dim; ++i)
    {
        hash_vector.push_back(rand() % 2 == 0 ? 1 : -1);
    }
}

// Compute the hash value from the input feature vector
int LSHHashFunction::hash(const std::vector<int> &feature) const
{
    int result = 0;
    for (int i = 0; i < m_dim; ++i)
    {
        // Don't change this into if-else structure for CUDA
        result += feature[i] * hash_vector[i];
    }
    return result >= 0 ? 1 : 0;
}

LSHCalculator::LSHCalculator(int num_hashes, int dim)
{
    // Set the number of hash functions needed
    m_num_hashes = num_hashes;

    // Initialize a batch of hash functions
    for (int i = 0; i < num_hashes; ++i)
    {
        hash_functions.push_back(LSHHashFunction(dim));
    }
}

/******************* LSHCalculator ***********************/

// Compute the hash values of an input feature vector
// Different hash values come from different hash funtions
std::vector<int> LSHCalculator::computeHash(const std::vector<int> &feature) const
{
    std::vector<int> hash_values;
    for (const auto &hash_function : hash_functions)
    {
        hash_values.push_back(hash_function.hash(feature));
    }
    return hash_values;
}

// Compute the similarity between two input feature vectors
double LSHCalculator::calculateSimilarity(const std::vector<int> &hash1, const std::vector<int> &hash2) const
{
    int common_hashes = 0;
    for (int i = 0; i < m_num_hashes; ++i)
    {
        if (hash1[i] == hash2[i])
        {
            common_hashes++;
        }
    }

    // Jaccard similarity coefficient
    double sim = static_cast<double>(common_hashes) / static_cast<double>(m_num_hashes);

    return sim;
}