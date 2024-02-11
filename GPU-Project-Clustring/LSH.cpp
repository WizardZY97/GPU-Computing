#include "LSH.h"

/******************* LSHHashFunction ***********************/

LSHHashFunction::LSHHashFunction(int dim, int order)
{
    // Set the dimension of the hash function
    m_dim = dim;

    // Initialize a hash function
    srand(time(nullptr) + order);
    for (int i = 0; i < dim; ++i)
    {
        hash_vector.push_back(rand() % 2 == 0 ? 1 : -1);
    }
}

// Compute the hash value from the input feature vector
int LSHHashFunction::hash(const std::vector<int> &feature) const
{
    int result = 0;
    for (unsigned int i = 0; i < feature.size(); ++i)
    {
        // Don't change this into if-else structure for CUDA
        result += feature[i] * hash_vector[i];
    }
    return result;
}

LSHCalculator::LSHCalculator(int num_hashes, int dim)
{
    // Set the number of hash functions needed
    m_num_hashes = num_hashes;

    // Initialize a batch of hash functions
    for (int i = 0; i < num_hashes; ++i)
    {
        hash_functions.push_back(LSHHashFunction(dim, i));
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
    double dot_product = 0, norm_hash1 = 0, norm_hash2 = 0;
    for (int i = 0; i < m_num_hashes; ++i)
    {
        double d_hash1 = static_cast<double>(hash1[i]);
        double d_hash2 = static_cast<double>(hash2[i]);

        dot_product += d_hash1 * d_hash2;
        norm_hash1 += d_hash1 * d_hash1;
        norm_hash2 += d_hash2 * d_hash2;
    }
    norm_hash1 = sqrt(norm_hash1);
    norm_hash2 = sqrt(norm_hash2);

    // Cosine similarity coefficient
    double sim = dot_product / (norm_hash1 * norm_hash2);

    return sim;
}