#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000

__device__ double f(double x) {
    return x * x;
}


__global__ void integrate(double a, double h, double *partial_sum, int mode) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (mode == 0) {

        if (tid < N) {
            double x1 = a + tid * h;
            double x2 = a + (tid + 1) * h;
            partial_sum[tid] = (f(x1) + f(x2)) * h / 2.0;
        }
    }
    else {

        for (int i = tid; i < N; i += stride) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            partial_sum[i] = (f(x1) + f(x2)) * h / 2.0;
        }
    }
}

// -------------------- REDUCTION KERNEL --------------------
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

// -------------------- MAIN --------------------
int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    double *h_partial = new double[N];
    double *d_partial, *d_temp;

    cudaMalloc(&d_partial, N * sizeof(double));
    cudaMalloc(&d_temp, N * sizeof(double));

    int threadsPerBlock = 256;

    for (int mode = 0; mode <= 1; mode++) {

        int blocks = (mode == 0)
            ? ((N + threadsPerBlock - 1) / threadsPerBlock)
            : 1024;

        auto start = std::chrono::high_resolution_clock::now();

        // ---------------- GPU COMPUTE ----------------
        integrate<<<blocks, threadsPerBlock>>>(a, h, d_partial, mode);
        cudaDeviceSynchronize();

        // ---------------- GPU REDUCTION ----------------
        int n = N;
        double *in = d_partial;
        double *out = d_temp;

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

        std::cout << "\nMODE " << mode << " (Blocks: " << blocks << ")" << std::endl;
        std::cout << "Integral = " << result << std::endl;
        std::cout << "Time = "
                  << std::chrono::duration<double>(end - start).count()
                  << " sec" << std::endl;
    }

    cudaFree(d_partial);
    cudaFree(d_temp);
    delete[] h_partial;

    return 0;
}