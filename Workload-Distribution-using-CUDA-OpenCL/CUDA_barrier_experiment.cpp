#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

// ---------------- KERNEL WITHOUT / WITH BARRIER ----------------
__global__ void integrate_sync(double a, double h, double *partial_sum, int use_barrier) {

    __shared__ double cache[256]; // for 256 threads per block

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;

    double val = 0.0;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        val = (f(x1) + f(x2)) * h / 2.0;
    }

    cache[local] = val;

    if (use_barrier) {
        __syncthreads();
    }

    // simple write-back
    if (tid < N) {
        partial_sum[tid] = cache[local];
    }
}

// ---------------- MAIN ----------------
int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    double *h_partial = new double[N];
    double *d_partial;

    cudaMalloc(&d_partial, N * sizeof(double));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int use_barrier = 0; use_barrier <= 1; use_barrier++) {

        auto start = std::chrono::high_resolution_clock::now();

        integrate_sync<<<blocks, threadsPerBlock>>>(
            a, h, d_partial, use_barrier
        );

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_partial, d_partial, N * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += h_partial[i];
        }

        

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "\nBARRIER = " << use_barrier << std::endl;
        std::cout << "Time = " << elapsed.count() << " sec" << std::endl;
    }

    cudaFree(d_partial);
    delete[] h_partial;

    return 0;
}