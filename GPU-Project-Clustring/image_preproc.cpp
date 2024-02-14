#include "image_preproc.h"

void readImageToArr(const char* filename, int* arr)
{
    // Read the image from the given file name
    cimg_library::CImg<unsigned char> image(filename);

    // Convert the RGB image into a gray one
    cimg_library::CImg<unsigned char> grayImage = image.get_RGBtoYCbCr().channel(0);

    // Resize to 1024*1024
    cimg_library::CImg<unsigned char> resized_image = grayImage.resize(SIZE, SIZE);

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            arr[i * SIZE + j] = *resized_image.data(j, i, 0, 0);
        }
    }
    
}

std::vector<std::vector<int>> readImageToVec(const char* filename) 
{
    // Read the image from the given file name
    cimg_library::CImg<unsigned char> image(filename);

    // Convert the RGB image into a gray one
    cimg_library::CImg<unsigned char> grayImage = image.get_RGBtoYCbCr().channel(0);

    // Resize to 1024*1024
    int target_width = SIZE, target_height = SIZE;
    cimg_library::CImg<unsigned char> resized_image = grayImage.resize(target_width, target_height);

    std::vector<std::vector<int>> vec_image(target_height, std::vector(target_width, 0));
    for (int i = 0; i < target_height; i++)
    {   
        for (int j = 0; j < target_width; j++)
        {
            vec_image[i][j] = *resized_image.data(j, i, 0, 0);
        }
    }

    return vec_image;
}

std::vector<std::vector<int>> applySobel(const std::vector<std::vector<int>>& inputImage) 
{
    int rows = inputImage.size();
    int cols = inputImage[0].size();

    std::vector<std::vector<int>> result(rows, std::vector<int>(cols, 0));
    
    // Define the Sobel kernels (calculators) on the axe X, Y
    std::vector<std::vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    std::vector<std::vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int gx = 0, gy = 0;

            // Calculate the gradient of pixel(i, j) on the axe X, Y
            for (int u = -1; u <= 1; ++u) {
                for (int v = -1; v <= 1; ++v) {
                    gx += inputImage[i + u][j + v] * sobelX[u + 1][v + 1];
                    gy += inputImage[i + u][j + v] * sobelY[u + 1][v + 1];
                }
            }

            // Calculate the value of gradient
            result[i][j] = static_cast<int>(sqrt(gx * gx + gy * gy));
        }
    }

    return result;
}

// Turn the 2D vector into 1D
std::vector<int> flatten(const std::vector<std::vector<int>>& originalMat)
{
    int rows = originalMat.size();
    int cols = originalMat[0].size();

    int vecSize = rows * cols;

    std::vector<int> vec(vecSize, 0);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            vec[i * cols + j] = originalMat[i][j];
        }
    }
    
    return vec;
}