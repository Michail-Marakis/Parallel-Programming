#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000
#define BLOCK_SIZE 256

__device__ double f(double x) {
    return x * x;
}


//REGISTER CASE

__global__ void kernel_registers(double a, double h, double *out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        double val = (f(x1) + f(x2)) * h / 2.0;
        out[tid] = val;
    }
}


//GLOBAL CASE

__global__ void kernel_global(double a, double h, double *out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        out[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}


//SHARED 

__global__ void kernel_shared(double a, double h, double *partial, int n_elements) {

    __shared__ double cache[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    if (i < n_elements) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        value = (f(x1) + f(x2)) * h * 0.5;
    }

    cache[tid] = value;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = cache[0];
    }
}


//REDUCTION KERNEL (FULL GPU FINAL SUM)

__global__ void reduce(double *input, double *output, int n) {

    __shared__ double cache[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;

    double val = (tid < n) ? input[tid] : 0.0;

    cache[local] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local < s) {
            cache[local] += cache[local + s];
        }
        __syncthreads();
    }

    if (local == 0) {
        output[blockIdx.x] = cache[0];
    }
}


//FULL GPU REDUCTION PIPELINE

double gpu_reduce(double *d_in, double *d_tmp, int n, int threads) {

    double *in = d_in;
    double *out = d_tmp;

    while (n > 1) {

        int blocks = (n + threads - 1) / threads;

        reduce<<<blocks, threads>>>(in, out, n);
        cudaDeviceSynchronize();

        n = blocks;

        double *tmp = in;
        in = out;
        out = tmp;
    }

    double result;
    cudaMemcpy(&result, in, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}


//MAIN

int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int threadsPerBlock = BLOCK_SIZE;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    double *h_out_N = new double[N];
    double *d_out_N, *d_tmp;

    cudaMalloc(&d_out_N, N * sizeof(double));
    cudaMalloc(&d_tmp, N * sizeof(double));

    std::cout << "--- CUDA MEMORY MODELS (FULL GPU REDUCTION) ---\n";


    //REGISTER
    
    auto start = std::chrono::high_resolution_clock::now();

    kernel_registers<<<blocks, threadsPerBlock>>>(a, h, d_out_N);
    cudaDeviceSynchronize();

    double sum_reg = gpu_reduce(d_out_N, d_tmp, N, threadsPerBlock);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "\n1. REGISTER CASE -> " << sum_reg
              << " | Time = "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

  
    //GLOBAL

    start = std::chrono::high_resolution_clock::now();

    kernel_global<<<blocks, threadsPerBlock>>>(a, h, d_out_N);
    cudaDeviceSynchronize();

    double sum_glob = gpu_reduce(d_out_N, d_tmp, N, threadsPerBlock);

    end = std::chrono::high_resolution_clock::now();

    std::cout << "\n2. GLOBAL CASE -> " << sum_glob
              << " | Time = "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";


    //SHARED

    double *d_blocks;
    cudaMalloc(&d_blocks, blocks * sizeof(double));

    start = std::chrono::high_resolution_clock::now();

    kernel_shared<<<blocks, threadsPerBlock>>>(a, h, d_blocks, N);
    cudaDeviceSynchronize();

    double sum_shared = gpu_reduce(d_blocks, d_tmp, blocks, threadsPerBlock);

    end = std::chrono::high_resolution_clock::now();

    std::cout << "\n3. SHARED CASE -> " << sum_shared
              << " | Time = "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    cudaFree(d_out_N);
    cudaFree(d_tmp);
    cudaFree(d_blocks);

    delete[] h_out_N;

    return 0;
}