//
// Created by qin on 23/11/23.
//

#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

// Kernel function to implement Sobel edge detection
__global__ void SobelGPU(unsigned char *in, unsigned char *out, int height, int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * width + x;
    int Gx = 0;
    int Gy = 0;
    unsigned char x0, x1, x2, x3, x5, x6, x7, x8;
    if (x > 0 && x < width && y > 0 && y < height) {
        x0 = in[(y - 1) * width + x - 1];
        x1 = in[(y - 1) * width + x];
        x2 = in[(y - 1) * width + x + 1];
        x3 = in[(y) * width + x - 1];
        x5 = in[(y) * width + x + 1];
        x6 = in[(y + 1) * width + x - 1];
        x7 = in[(y + 1) * width + x];
        x8 = in[(y + 1) * width + x + 1];
        // horizontal convolution kernel
        Gx = (x0 + x3 * 2 + x6) - (x2 + x5 * 2 + x8);
        // vertical convolution kernel
        Gy = (x0 + x1 * 2 + x2) - (x6 + x7 * 2 + x8);
    }
    out[index] = (abs(Gx) + abs(Gy)) / 2;
}

int main(void) {
    cv::Mat input_image = imread("./Lenna_original.jpg", cv::IMREAD_GRAYSCALE);
    // assign the memory to store the output image
    cv::Mat output_image(input_image.rows, input_image.cols, CV_8UC1, cv::Scalar(0));

    size_t num_pixels = input_image.rows * input_image.cols * sizeof(unsigned char);
    unsigned char *in_gpu, *out_gpu;
    cudaMalloc(&in_gpu, num_pixels);
    cudaMalloc(&out_gpu, num_pixels);

    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid((input_image.cols + threads_per_block.x - 1) / threads_per_block.x,
                         (input_image.rows + threads_per_block.y - 1) / threads_per_block.y);

    clock_t start_time, end_time;
    start_time = clock();

//  // Sobel without streams
//  cudaMemcpy(in_gpu, input_image.data, num_pixels, cudaMemcpyHostToDevice);
//  SobelGPU<<<blocks_per_grid, threads_per_block>>>
//      (in_gpu, out_gpu, input_image.rows, input_image.cols);
//  cudaMemcpy(output_image.data, out_gpu, num_pixels, cudaMemcpyDeviceToHost);

    // Sobel using streams
    int num_streams = 16;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    int chunk_size = num_pixels / num_streams;
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        // copy data from host to device
        cudaMemcpyAsync(in_gpu + offset, input_image.data + offset, chunk_size, cudaMemcpyHostToDevice, streams[i]);
        SobelGPU<<<blocks_per_grid, threads_per_block, 0, streams[i]>>>
                (in_gpu + offset, out_gpu + offset, input_image.rows, input_image.cols);
        cudaMemcpyAsync(output_image.data + offset, out_gpu + offset, chunk_size, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();
    end_time = clock();
    std::cout << "Totle Time : " << (double) (end_time - start_time) << std::endl;

    imshow("input", input_image);
    imshow("output", output_image);
    cv::waitKey(0);

    // Free memory
    cudaFree(in_gpu);
    cudaFree(out_gpu);

    return 0;
}