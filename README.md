C++ Cuda project
A c++ program that utilizes a cuda kernel to apply a box filter to a png file.
I use the STBI_IMAGE library to load the image files and extract the pixel rgb data.
Using cuda to apply the averaging box blue filter with parallelization to exploit the larger amount of cores available on the gpu vs the cpu.

Original image
![spaghetti](https://github.com/user-attachments/assets/33732734-a9b2-4b44-b831-9cc2ace72ec5)

Resultant image from program output

![box_blur_image](https://github.com/user-attachments/assets/20ce478a-2eaf-4bb5-8099-b74c0e78c572)
