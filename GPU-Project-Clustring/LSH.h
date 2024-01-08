#ifndef __LSH_H__
#define __LSH_H__

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// LSH Hash function
class LSHHashFunction
{
private:
    int m_dim;                    // Dimension of the hash function
                                  // (depends on the size of target feature vector)
    std::vector<int> hash_vector; // Hash function

public:
    LSHHashFunction(int dim, int order);

    // Compute the hash value from the input feature vector
    int hash(const std::vector<int> &feature) const;
};

// The calculator for LSH
class LSHCalculator
{
private:
    int m_num_hashes;                            // Number of the hash functions needed
                                                 // (pre-set by the developper)
    std::vector<LSHHashFunction> hash_functions; // Hash functions needed

public:
    LSHCalculator(int num_hashes, int dim);

    // Compute the hash values of an input feature vector
    // Different hash values come from different hash funtions
    std::vector<int> computeHash(const std::vector<int> &feature) const;

    // Compute the similarity between two input feature vectors
    double calculateSimilarity(const std::vector<int> &hash1, const std::vector<int> &hash2) const;
};

#endif