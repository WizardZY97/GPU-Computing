#ifndef __IMAGE_PREPROC_H__
#define __IMAGE_PREPROC_H__

#define cimg_imagemagick_path "/user/3/wuzho/Desktop/MOSIG-M2/GPU-Computing/GPU-Project-Clustring/magick"
#include "CImg/CImg.h"
#include <cmath>
#include <vector>

#define SIZE 1024

void readImageToArr(const char* filename, int* arr);

std::vector<std::vector<int>> readImageToVec(const char* filename);

std::vector<std::vector<int>> applySobel(const std::vector<std::vector<int>>& inputImage);

std::vector<int> flatten(const std::vector<std::vector<int>>& originalMat);

void saveImage(const std::vector<std::vector<int>>& vec_image, const char* filename);

#endif