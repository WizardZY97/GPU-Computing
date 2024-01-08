#include "image_preproc.h"

std::vector<std::vector<int>> readImageToVec(const char* filename) 
{
    // Read the image from the given file name
    cimg_library::CImg<unsigned char> image(filename);

    // Convert the RGB image into a gray one
    cimg_library::CImg<unsigned char> grayImage = image.get_RGBtoYCbCr().channel(0);

    // Resize to 512x512
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

void saveImage(const std::vector<std::vector<int>>& vec_image, const char* filename)
{
    // 获取图像的宽度和高度
    const unsigned int width = vec_image[0].size();
    const unsigned int height = vec_image.size();

    // 创建CImg对象
    cimg_library::CImg<int> image(width, height, 1, 1, 0);

    // 将二维vector的数据复制到CImg对象中
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            image(x, y) = vec_image[y][x];
        }
    }

    image.save(filename);
}