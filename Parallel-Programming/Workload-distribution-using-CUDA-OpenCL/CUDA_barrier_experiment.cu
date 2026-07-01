#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000 // πλήθος διαστημάτων

__device__ double f(double x) {
    return x * x;
}


//kernel υπολογισμού τραπεζοειδών

__global__ void compute_kernel(double a, double h, double *out) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x; //global id

    if (tid < N) { //bound check
        double x1 = a + tid * h;       //αριστερό σημείο
        double x2 = a + (tid + 1) * h; //δεξί σημείο

        out[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}

//reduction kernel
__global__ void reduce_kernel(double *input, double *output, int n, int use_barrier) {

    __shared__ double cache[256]; // shared memory block

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global id
    int local = threadIdx.x; // local id

    // φόρτωμα δεδομένων
    double val = (tid < n) ? input[tid] : 0.0;

    cache[local] = val; // write στο shared

    if (use_barrier) {
        __syncthreads(); // sync threads
    }

    // reduction σε shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {

        if (local < s) {
            cache[local] += cache[local + s]; // άθροιση
        }

        if (use_barrier) {
            __syncthreads(); // sync κάθε βήμα
        }
    }

    // thread 0 γράφει το αποτέλεσμα
    if (local == 0) {
        output[blockIdx.x] = cache[0];
    }
}


//main
int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N; //βήμα ολοκλήρωσης

    int threads = 256; //threads per block
    int blocks = (N + threads - 1) / threads; //blocks

    double *d_in, *d_out;

    cudaMalloc(&d_in, N * sizeof(double));       //input buffer
    cudaMalloc(&d_out, blocks * sizeof(double)); //output buffer

    double *h_out = new double[blocks]; //host buffer

    std::cout << "\n--- CUDA BARRIER vs NO BARRIER ---\n";

    //δοκιμή με και χωρίς barrier
    for (int use_barrier = 0; use_barrier <= 1; use_barrier++) {

        auto start = std::chrono::high_resolution_clock::now();

        //step 1: υπολογισμός
        compute_kernel<<<blocks, threads>>>(a, h, d_in);
        cudaDeviceSynchronize();

        //step 2: reduction πλήρως στο GPU
        int n = N;
        double *in = d_in;
        double *out = d_out;

        while (n > 1) {

            int b = (n + threads - 1) / threads; //νέα blocks

            reduce_kernel<<<b, threads>>>(in, out, n, use_barrier);
            cudaDeviceSynchronize();

            n = b; //νέο μέγεθος

            //swap pointers
            double *tmp = in;
            in = out;
            out = tmp;
        }

        double result;

        //τελικό αποτέλεσμα στη CPU
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