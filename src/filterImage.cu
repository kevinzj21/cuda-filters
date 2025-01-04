#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define BLOCK_SIZE (16u)
#define FILTER_SIZE (5u)
#define TILE_SIZE (12U) // BLOCK_SIZE - (2 * (FILTER_SIZE/2))


//Compile with: nvcc -o filterImage src/filterImage.cu -I./src

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

//define kernel function
__global__ void processImage(unsigned char * out, const unsigned char * in, size_t pitch, unsigned int width, unsigned int height){

    int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

    //Indexing the surrounding pixels based on the chosen filter size
    int x_i = x_o - FILTER_SIZE/2;
    int y_i = y_o - FILTER_SIZE/2;

    int sum = 0;

    //defining shared memory which is a 2d array
    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    //copying to shared memory only if pixel is within the image
    if ((x_i >= 0) && (x_i < width) && (y_i >=0) && (y_i < height)){
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
    }
    else
        sBuffer[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();
    //iterate through the filter and add up all the colour values of the pixels
    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE){
        for (int r = 0; r < FILTER_SIZE; r++)
            for (int c = 0; c < FILTER_SIZE; c++)
                sum += sBuffer[threadIdx.y + r][threadIdx.x + c];
        
        //averaging colour values then writing result
        sum /= FILTER_SIZE * FILTER_SIZE;
        if (x_o < width && y_o < height)
            out[y_o * width + x_o] = sum;

    }

}

int main(int argc, char const *argv[])
{
    printf("Attempting to load image...");

    //Use stbi image library to load the image file
    int width, height, channels;
    unsigned char *img = stbi_load("spaghetti.PNG", &width, &height, &channels, 3);
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
    std::vector<unsigned char> rgb_val(img, img + (width * height * 3));
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            int index = (y * width + x) * 3;

            h_r[y * width + x] = rgb_val[index + 0];
            h_g[y * width + x] = rgb_val[index + 1];
            h_b[y * width + x] = rgb_val[index + 2];
        }
    }

    //Allocate the device memory for the result
    unsigned char *d_r_o = NULL;
    unsigned char *d_g_o = NULL;
    unsigned char *d_b_o = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_o, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_o, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_o, size));

    //Allocate device memory for rgb channels
    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;

    //Use a pitch variable to enable accurate burst reading of colour channels
    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;

    CUDA_CHECK_RETURN( cudaMallocPitch(&d_r, &pitch_r, width, height) );
    CUDA_CHECK_RETURN( cudaMallocPitch(&d_g, &pitch_g, width, height) );
    CUDA_CHECK_RETURN( cudaMallocPitch(&d_b, &pitch_b, width, height) );

    // Copy host data to device
    // The first width is the pitch of the host data which is just the row size
    CUDA_CHECK_RETURN( cudaMemcpy2D(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy2D(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice) );
    CUDA_CHECK_RETURN( cudaMemcpy2D(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice) );

    //Image Kernel block size config

    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    //Run the kernel on each colour channel of the image
    processImage<<<grid_size,block_size>>>(d_r_o, d_r, pitch_r, width, height);
    processImage<<<grid_size,block_size>>>(d_g_o, d_g, pitch_g, width, height);
    processImage<<<grid_size,block_size>>>(d_b_o, d_b, pitch_b, width, height);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    //Copy results back to host from the device
    CUDA_CHECK_RETURN( cudaMemcpy(h_r_o, d_r_o, size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK_RETURN( cudaMemcpy(h_g_o, d_g_o, size, cudaMemcpyDeviceToHost) );
    CUDA_CHECK_RETURN( cudaMemcpy(h_b_o, d_b_o, size, cudaMemcpyDeviceToHost) );

    //Turning resulting pixel data back into an image
    std::vector<unsigned char> rgb_data(width * height * 3);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            int index = (i * width + j) * 3;

            rgb_data[index + 0] = h_r_o[i * width + j];
            rgb_data[index + 1] = h_g_o[i * width + j];
            rgb_data[index + 2] = h_b_o[i * width + j];
        
        }
    }


    //create output path and save resultant image
    const char* output_path = "box_blur_image.png";
    stbi_write_png(output_path, width, height, 3, rgb_data.data(), width * 3);
    stbi_write_png("test.png", width, height, 3, rgb_val.data(), width * 3);
    

    //Free allocated memory once finished
    free(h_r);
    free(h_g);
    free(h_b);

    free(h_r_o);
    free(h_g_o);
    free(h_b_o);

    CUDA_CHECK_RETURN(cudaFree(d_r_o));
    CUDA_CHECK_RETURN(cudaFree(d_g_o));
    CUDA_CHECK_RETURN(cudaFree(d_b_o));

    
    stbi_image_free(img);

    return 0;
}
