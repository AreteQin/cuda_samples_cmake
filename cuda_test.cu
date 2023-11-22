// includes, system
#include <stdio.h>
#include <ctime>
#include <iostream>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void increment_kernel(int *g_data, int inc_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x) {
    for (int i = 0; i < n; i++)
        if (data[i] != x) {
            printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
            return false;
        }

    return true;
}

int main(int argc, char *argv[]) {
    int devID = 0;
    cudaDeviceProp deviceProps;

    printf("[%s] - Starting...\n", argv[0]);

    // get device name
    cudaGetDeviceProperties(&deviceProps, devID);
    printf("CUDA device [%s]\n", deviceProps.name);

    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value = 26;

    // allocate host memory
    int *a = 0;
    cudaMallocHost((void **)&a, nbytes);
    memset(a, 0, nbytes);

    // allocate device memory
    int *d_a = 0;
    cudaMalloc((void **)&d_a, nbytes);
    cudaMemset(d_a, 255, nbytes);

    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    cudaProfilerStart();
    // record start time
    clock_t start_time = clock();
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(stop, 0);
    // record end time
    clock_t end_time = clock();
    cudaProfilerStop();

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    cudaEventElapsedTime(&gpu_time, start, stop);

    // print the cpu and gpu times
    printf("time spent executing by the GPU: %.2f\n", gpu_time);
    std::cout << "time spent by CPU in CUDA calls: " << ((float)(end_time - start_time)) / CLOCKS_PER_SEC << std::endl;
    printf("CPU executed %lu iterations while waiting for GPU to finish\n",
           counter);

    // check the output for correctness
    bool bFinalResults = correct_output(a, n, value);

    // release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(a);
    cudaFree(d_a);

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
