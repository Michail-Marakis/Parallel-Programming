#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000

// ---------------- DEVICE FUNCTION ----------------
__device__ double f(double x) {
    return x * x;
}

// ---------------- MAIN KERNEL ----------------
__global__ void integrate(double a, double h, double *out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        out[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}

// ---------------- REDUCTION KERNEL ----------------
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

    double *d_in, *d_tmp;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_tmp, N * sizeof(double));

    double *h_dummy = new double[N];

    int blockSizes[] = {32, 64, 128, 256, 512};

    std::cout << "\n--- CUDA BLOCK SIZE + FULL GPU REDUCTION ---\n";

    for (int bsize : blockSizes) {

        int threadsPerBlock = bsize;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        auto start = std::chrono::high_resolution_clock::now();

        // ---------------- STEP 1: compute ----------------
        integrate<<<blocks, threadsPerBlock>>>(a, h, d_in);
        cudaDeviceSynchronize();

        // ---------------- STEP 2: full GPU reduction ----------------
        int n = N;
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

        std::cout << "\nBLOCK SIZE = " << bsize << std::endl;
        std::cout << "Integral = " << result << std::endl;
        std::cout << "Time = "
                  << std::chrono::duration<double>(end - start).count()
                  << " sec" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_tmp);
    delete[] h_dummy;

    return 0;
}