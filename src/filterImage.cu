#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

//CUDA check for debugging
#define CUDA_CHECK_RETURN(call)                                          \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",          \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(int argc, char const *argv[])
{
    printf("Attempting to load image...");

    //Use stbi image library to load the image file
    int width, height, channels;
    unsigned char *img = stbi_load("spaghetti.PNG", &width, &height, &channels, 0);
    int size = width * height * sizeof(unsigned char);

    if(img == NULL){
        printf("Error in loading the image\n");
        exit(1);
    }
    //host memory for rgb values of the image
    unsigned char *h_r = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_g = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_b = (unsigned char*) malloc (size * sizeof(unsigned char));

    //memory allocation for output
    unsigned char *h_r_o = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_g_o = (unsigned char*) malloc (size * sizeof(unsigned char));
    unsigned char *h_b_o = (unsigned char*) malloc (size * sizeof(unsigned char));

    //processing image into rgb host memory
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            int index = (y * width + x) * channels;

            h_r[y * width + x] = img[index + 0];
            h_g[y * width + x] = img[index + 1];
            h_b[y * width + x] = img[index + 2];
        
        }
    }

    //Allocate the device memory for the result
    unsigned char *d_r_o = NULL;
    unsigned char *d_g_o = NULL;
    unsigned char *d_b_o = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_o, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_o, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_o, size));


    //Free memory once finished
    CUDA_CHECK_RETURN(cudaFree(d_r_o));
    CUDA_CHECK_RETURN(cudaFree(d_g_o));
    CUDA_CHECK_RETURN(cudaFree(d_b_o));
    free(h_r);
    free(h_g);
    free(h_b);
    free(h_r_o);
    free(h_r_o);
    free(h_r_o);
    stbi_image_free(img);

    return 0;
}
