#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000 


__device__ double f(double x) {
    return x * x;
}

//integrate kernel
//mode 0: 1 στοιχείο / thread
//mode 1: mult elem/thread
__global__ void integrate(double a, double h, double *partial_sum, int mode) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x; //global id
    int stride = blockDim.x * gridDim.x; //συνολικά threads

    if (mode == 0) {

        //thread -> element
        if (tid < N) {
            double x1 = a + tid * h;
            double x2 = a + (tid + 1) * h;

            partial_sum[tid] = (f(x1) + f(x2)) * h / 2.0;
        }
    }
    else {

        //grid-stride loop
        for (int i = tid; i < N; i += stride) {

            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;

            partial_sum[i] = (f(x1) + f(x2)) * h / 2.0;
        }
    }
}

//reduction kernel
__global__ void reduce(double *input, double *output, int n) {

    __shared__ double cache[256]; //shared memory

    int tid = blockIdx.x * blockDim.x + threadIdx.x; //global id
    int local = threadIdx.x; //local id

    //load με boundary check
    double val = (tid < n) ? input[tid] : 0.0;

    cache[local] = val;
    __syncthreads(); //sync πριν reduction

    //tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {

        if (local < s) {
            cache[local] += cache[local + s]; //combine
        }

        __syncthreads(); //sync κάθε βήμα
    }

    //write αποτέλεσμα block
    if (local == 0) {
        output[blockIdx.x] = cache[0];
    }
}

int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N; //βήμα

    double *h_partial = new double[N]; //host buffer (unused)
    double *d_partial, *d_temp;

    cudaMalloc(&d_partial, N * sizeof(double)); //main buffer
    cudaMalloc(&d_temp, N * sizeof(double));    //temp buffer

    int threadsPerBlock = 256;

    //mode 0 vs mode 1
    for (int mode = 0; mode <= 1; mode++) {

        //blocks επιλογή
        int blocks = (mode == 0)
            ? ((N + threadsPerBlock - 1) / threadsPerBlock) //full coverage
            : 1024; //fixed blocks για stride

        auto start = std::chrono::high_resolution_clock::now();

        //compute phase
        integrate<<<blocks, threadsPerBlock>>>(a, h, d_partial, mode);
        cudaDeviceSynchronize();

        //reduction phase
        int n = N;
        double *in = d_partial;
        double *out = d_temp;

        while (n > 1) {

            int b = (n + threadsPerBlock - 1) / threadsPerBlock; //νέα blocks

            reduce<<<b, threadsPerBlock>>>(in, out, n);
            cudaDeviceSynchronize();

            n = b; //νέο μέγεθος

            //swap buffers
            double *tmp = in;
            in = out;
            out = tmp;
        }

        double result;

        //copy τελικού αποτελέσματος
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