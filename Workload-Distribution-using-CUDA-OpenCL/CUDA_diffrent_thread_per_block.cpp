#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000

// -------------------- DEVICE FUNCTION --------------------
__device__ double f(double x) {
    return x * x;
}

// -------------------- KERNEL --------------------
__global__ void integrate(double a, double h, double *partial_sum) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        partial_sum[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}

// -------------------- MAIN --------------------
int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    double *h_partial = new double[N];
    double *d_partial;

    cudaMalloc(&d_partial, N * sizeof(double));

    
    int blockSizes[] = {32, 64, 128, 256, 512};

    for (int bsize : blockSizes) {

        int threadsPerBlock = bsize;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        auto start = std::chrono::high_resolution_clock::now();

        integrate<<<blocks, threadsPerBlock>>>(a, h, d_partial);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_partial, d_partial, N * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += h_partial[i];
        }

        

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "\nBLOCK SIZE = " << bsize << std::endl;
        std::cout << "Integral = " << sum << std::endl;
        std::cout << "Time = " << elapsed.count() << " sec" << std::endl;
    }

    cudaFree(d_partial);
    delete[] h_partial;

    return 0;
}