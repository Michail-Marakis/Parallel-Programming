#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000


__device__ double f(double x) {
    return x * x;
}


//STRATEGY 1: MULTIPLE ELEMENTS PER THREAD (Grid-Stride)

__global__ void integrate_multi(double a, double h, double *out) {
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


//STRATEGY 2: ONE ELEMENT PER THREAD

__global__ void integrate_one(double a, double h, double *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        out[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}

/
//REDUCTION KERNEL

__global__ void reduce(double *input, double *output, int n) {
    extern __shared__ double cache[]; 

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


void run_experiment(const char* label, int blocks, int threadsPerBlock, int starting_n, double a, double h, int use_multi_kernel, double *d_in, double *d_tmp) {
    auto start = std::chrono::high_resolution_clock::now();

    //Υπολογισμός Ολοκληρώματος
    if (use_multi_kernel) {
        integrate_multi<<<blocks, threadsPerBlock>>>(a, h, d_in);
    } else {
        integrate_one<<<blocks, threadsPerBlock>>>(a, h, d_in);
    }
    cudaDeviceSynchronize();

    //Full GPU Reduction
    int n = starting_n;
    double *in = d_in;
    double *out = d_tmp;

    while (n > 1) {
        int b = (n + threadsPerBlock - 1) / threadsPerBlock;
        reduce<<<b, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(in, out, n);
        cudaDeviceSynchronize();

        n = b;
        double *tmp = in;
        in = out;
        out = tmp;
    }

    double result;
    cudaMemcpy(&result, in, sizeof(double), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "  " << label << " -> Integral = " << result 
              << " | Time = " << std::chrono::duration<double>(end - start).count() << " sec\n";
}

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int blockSizes[] = {32, 64, 128, 256, 512};
    int static_blocks = 1024; //Σταθερά blocks για το Multi-element

    std::cout << "--- CUDA COMPLETE EXPERIMENT (ALL BLOCK SIZES & WORKLOADS) ---\n";
    std::cout << "N = " << N << "\n";

    for (int bsize : blockSizes) {
        std::cout << "\n=========================================\n";
        std::cout << " TESTING WITH BLOCK SIZE = " << bsize;
        std::cout << "\n=========================================\n";

        //MULTIPLE ELEMENTS PER THREAD
        int totalThreadsMulti = static_blocks * bsize;
        double *d_in_m, *d_tmp_m;
        cudaMalloc(&d_in_m, totalThreadsMulti * sizeof(double));
        cudaMalloc(&d_tmp_m, totalThreadsMulti * sizeof(double));

        run_experiment("Multi Elements/Thread", static_blocks, bsize, totalThreadsMulti, a, h, 1, d_in_m, d_tmp_m);

        cudaFree(d_in_m);
        cudaFree(d_tmp_m);

        //ONE ELEMENT PER THREAD
        int blocksOne = (N + bsize - 1) / bsize; 
        double *d_in_o, *d_tmp_o;
        
        //Εδώ χρειαζόμαστε buffer μεγέθους N επειδή παράγονται N αρχικά στοιχεία
        cudaMalloc(&d_in_o, N * sizeof(double));
        cudaMalloc(&d_tmp_o, blocksOne * sizeof(double));

        run_experiment("One Element/Thread   ", blocksOne, bsize, N, a, h, 0, d_in_o, d_tmp_o);

        cudaFree(d_in_o);
        cudaFree(d_tmp_o);
    }

    return 0;
}