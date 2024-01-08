#include "CImg/CImg.h"

int main() {
    // Read the image
    cimg_library::CImg<unsigned char> image("a.jpg");

    // Convert the RGB image into Gray image
    cimg_library::CImg<unsigned char> grayImage = image.get_RGBtoYCbCr().channel(0);

    // Resize to 32x32
    int target_width = 1024, target_height = 1024;
    cimg_library::CImg<unsigned char> resized_image = grayImage.resize(target_width, target_height);

    // Get the data of image
    // unsigned char *data = resized_image.data();

    printf("%d\n", *resized_image.data(0, 0, 0, 0));

    FILE *file = fopen("./result.txt", "w");
    for (int i = 0; i < target_height; i++)
    {   
        for (int j = 0; j < target_width; j++)
            fprintf(file, "%d ", *resized_image.data(j, i, 0, 0));
        fprintf(file, "\n");
    }
    fclose(file);

    // Save results
    resized_image.save("output_image.jpg");

    // Display
    resized_image.display();

    return 0;
}
