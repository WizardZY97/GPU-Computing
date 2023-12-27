#include "CImg/CImg.h"

int main() {
    cimg_library::CImg<unsigned char> image("a.jpg");

    // Resize to 32x32
    int target_width = 32, target_height = 32;
    cimg_library::CImg<unsigned char> resized_image = image.resize(target_width, target_height);

    // Display
    resized_image.display();

    return 0;
}
