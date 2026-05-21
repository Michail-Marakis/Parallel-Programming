#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}

// =======================================================
// STEP 1: compute partial values (same for both cases)
// =======================================================
__global__ void compute_kernel(double a, double h, double *out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        out[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}

// =======================================================
// STEP 2: REDUCTION WITH / WITHOUT BARRIER
// =======================================================
__global__ void reduce_kernel(double *input, double *output, int n, int use_barrier) {

    __shared__ double cache[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local = threadIdx.x;

    double val = (tid < n) ? input[tid] : 0.0;

    cache[local] = val;

    if (use_barrier) {
        __syncthreads();
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {

        if (local < s) {
            cache[local] += cache[local + s];
        }

        if (use_barrier) {
            __syncthreads();
        }
    }

    if (local == 0) {
        output[blockIdx.x] = cache[0];
    }
}

// =======================================================
// MAIN
// =======================================================
int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    double *d_in, *d_out;

    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, blocks * sizeof(double));

    double *h_out = new double[blocks];

    std::cout << "\n--- CUDA BARRIER vs NO BARRIER (FULL GPU REDUCTION) ---\n";

    for (int use_barrier = 0; use_barrier <= 1; use_barrier++) {

        auto start = std::chrono::high_resolution_clock::now();

        // STEP 1: compute
        compute_kernel<<<blocks, threads>>>(a, h, d_in);
        cudaDeviceSynchronize();

        // STEP 2: reduce (FULL GPU)
        int n = N;
        double *in = d_in;
        double *out = d_out;

        while (n > 1) {

            int b = (n + threads - 1) / threads;

            reduce_kernel<<<b, threads>>>(in, out, n, use_barrier);
            cudaDeviceSynchronize();

            n = b;

            // swap buffers
            double *tmp = in;
            in = out;
            out = tmp;
        }

        double result;
        cudaMemcpy(&result, in, sizeof(double), cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "\nBARRIER = " << use_barrier << std::endl;
        std::cout << "Result  = " << result << std::endl;
        std::cout << "Time    = "
                  << std::chrono::duration<double>(end - start).count()
                  << " sec" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_out;

    return 0;
}