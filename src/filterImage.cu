#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main(int argc, char const *argv[])
{
    printf("Attempting to load image...");
    int width, height, channels;
    unsigned char *img = stbi_load("spaghetti.PNG", &width, &height, &channels, 0);

    if(img == NULL){
        printf("Error in loading the image\n");
        exit(1);
    }

    stbi_write_png("test.png", width, height, channels, img, width * channels);
    return 0;
}
