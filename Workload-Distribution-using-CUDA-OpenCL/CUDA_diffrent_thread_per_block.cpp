#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000

// ---------------- DEVICE FUNCTION ----------------
__device__ double f(double x) {
    return x * x;
}

// ---------------- KERNEL 1 ----------------
// MULTIPLE ELEMENTS PER THREAD
__global__ void integrate(double a, double h, double *out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;

    for (int i = tid; i < N; i += stride) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        sum += (f(x1) + f(x2)) * h / 2.0;
    }

    out[tid] = sum;
}

// ---------------- KERNEL 2 ----------------
// REDUCTION
__global__ void reduce(double *input, double *output, int n) {

    __shared__ double cache[256];

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

// ---------------- MAIN ----------------
int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int threadsPerBlock = 256;

    int blocks = 1024;  

    int totalThreads = blocks * threadsPerBlock;

    double *d_in, *d_tmp;
    cudaMalloc(&d_in, totalThreads * sizeof(double));
    cudaMalloc(&d_tmp, totalThreads * sizeof(double));

    std::cout << "\n--- CUDA MULTI-ELEMENT PER THREAD + FULL GPU REDUCTION ---\n";

    auto start = std::chrono::high_resolution_clock::now();


    integrate<<<blocks, threadsPerBlock>>>(a, h, d_in);

    // 🔴 GLOBAL BARRIER
    cudaDeviceSynchronize();


    int n = totalThreads;
    double *in = d_in;
    double *out = d_tmp;

    while (n > 1) {

        int b = (n + threadsPerBlock - 1) / threadsPerBlock;

        reduce<<<b, threadsPerBlock>>>(in, out, n);

        cudaDeviceSynchronize();

        n = b;

        double *tmp = in;
        in = out;
        out = tmp;
    }

    double result;
    cudaMemcpy(&result, in, sizeof(double), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Integral = " << result << std::endl;
    std::cout << "Time = "
              << std::chrono::duration<double>(end - start).count()
              << " sec" << std::endl;

    cudaFree(d_in);
    cudaFree(d_tmp);

    return 0;
}