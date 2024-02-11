#ifndef __IMAGE_PREPROC_H__
#define __IMAGE_PREPROC_H__

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