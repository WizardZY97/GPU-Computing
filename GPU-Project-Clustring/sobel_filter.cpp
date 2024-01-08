#include <iostream>
#include <cmath>
#include <vector>


// 读取灰度图像
std::vector<std::vector<int>> readImage(const char* filename) {
    // 这里假设图像是灰度图，每个像素用一个整数表示
    // 实际场景中，你可能需要根据图像格式自己实现图像读取
    // 在这个示例中，我们使用二维数组表示图像
    // 注意：这是一个简化的例子，对于不同图像格式可能需要不同的处理
    // 在实际应用中，你可能需要使用专门的图像处理库或编写更复杂的图像读取代码
    // 以下代码仅用于示例目的
    std::vector<std::vector<int>> image;
    // 在这里添加图像读取代码...

    return image;
}

// 使用 Sobel 算子进行边缘检测
std::vector<std::vector<int>> applySobel(const std::vector<std::vector<int>>& inputImage) {
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

// 显示图像
void displayImage(const std::vector<std::vector<int>>& image) {
    // 这里添加显示图像的代码，具体实现取决于你使用的图形库
    // 在这个示例中，我们简化为在控制台打印灰度值
    for (const auto& row : image) {
        for (int pixel : row) {
            std::cout << pixel << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // 读取图像
    std::vector<std::vector<int>> inputImage = readImage("input_image.txt");

    // 使用 Sobel 算子进行边缘检测
    std::vector<std::vector<int>> edgeDetectedImage = applySobel(inputImage);

    // 显示原图和边缘检测结果
    std::cout << "Original Image:\n";
    displayImage(inputImage);

    std::cout << "\nEdge Detected Image:\n";
    displayImage(edgeDetectedImage);

    return 0;
}
