#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>
using namespace std;

std::string sha256(const std::string& input)
{
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input.c_str(), input.length());
    SHA256_Final(hash, &sha256);

    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }

    return ss.str();
}

std::string readFile(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    std::ostringstream content;
    content << file.rdbuf();

    return content.str();
}

int main(int argc, char *argv[])
{
    std::string imagePath = argv[1];

    std::string imageData = readFile(imagePath);

    std::string hashValue = sha256(imageData);

    std::cout << "SHA-256: " << hashValue << std::endl;

    return 0;
}