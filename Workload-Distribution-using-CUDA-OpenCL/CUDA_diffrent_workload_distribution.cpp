#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000

// -------------------- DEVICE FUNCTION --------------------
__device__ double f(double x) {
    return x * x;
}

//mode = 0 → 1 element per thread
//mode = 1 → multiple elements per thread

__global__ void integrate(double a, double h, double *partial_sum, int mode) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //-------- MODE 0: 1 element / thread ----------------
    if (mode == 0) {

        if (tid < N) {
            double x1 = a + tid * h;
            double x2 = a + (tid + 1) * h;
            partial_sum[tid] = (f(x1) + f(x2)) * h / 2.0;
        }
    }

    //------------ MODE 1: many elements / thread ----------------
    else {

        for (int i = tid; i < N; i += stride) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            partial_sum[i] = (f(x1) + f(x2)) * h / 2.0;
        }
    }
}

// -------------------- MAIN --------------------
int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    double *h_partial = new double[N];
    double *d_partial;
    cudaMalloc(&d_partial, N * sizeof(double));

    int threadsPerBlock = 256;

    for (int mode = 0; mode <= 1; mode++) {
        // Προσαρμογή των blocks ανάλογα με το μοντέλο κατανομής
        int blocks = (mode == 0) ? ((N + threadsPerBlock - 1) / threadsPerBlock) : 1024;

        auto start = std::chrono::high_resolution_clock::now();

        integrate<<<blocks, threadsPerBlock>>>(a, h, d_partial, mode);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(h_partial, d_partial, N * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += h_partial[i];
        }

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\nMODE " << mode << " (Blocks: " << blocks << ")" << std::endl;
        std::cout << "Integral = " << sum << std::endl;
        std::cout << "Time = " << elapsed.count() << " sec" << std::endl;
    }

    cudaFree(d_partial);
    delete[] h_partial;
    return 0;
}