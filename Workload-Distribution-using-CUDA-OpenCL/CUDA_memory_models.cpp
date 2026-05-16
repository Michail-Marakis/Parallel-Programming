#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define N 100000000 // 10^8 υποδιαιρέσεις
#define BLOCK_SIZE 256 // Σταθερό μέγεθος block όπως συνηθίζεται στις διαφάνειες

// -------------------- DEVICE FUNCTION --------------------
__device__ double f(double x) {
    return x * x; // f(x) = x^2
}

// =======================================================
// CASE 1: REGISTER / LOCAL VARIABLES (Writes N elements)
// =======================================================
__global__ void kernel_registers(double a, double h, double *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;

        // Χρήση τοπικής μεταβλητής (καταχωρητής) πριν την εγγραφή
        double val = (f(x1) + f(x2)) * h / 2.0;
        out[tid] = val;
    }
}

// =======================================================
// CASE 2: GLOBAL MEMORY ONLY (Writes N elements directly)
// =======================================================
__global__ void kernel_global(double a, double h, double *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;

        // Απευθείας εγγραφή του αποτελέσματος στην Global Memory
        out[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}

// =======================================================
// CASE 3: SHARED MEMORY + REDUCTION (Writes 'blocks' elements)
// =======================================================
__global__ void kernel_shared(double a, double h, double *partial, int n_elements) {
    // Στατική δήλωση shared μνήμης με βάση τη σταθερά BLOCK_SIZE
    __shared__ double cache[BLOCK_SIZE]; 

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;

    if (i < n_elements) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        value = (f(x1) + f(x2)) * h * 0.5;
    }

    cache[tid] = value;
    __syncthreads(); 

    // Παράλληλη αναγωγή (Reduction) μέσα στο block
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }
        __syncthreads(); 
    }

    // Το thread 0 κάθε block γράφει το μερικό άθροισμα στην global memory
    if (tid == 0) {
        partial[blockIdx.x] = cache[0];
    }
}

// =======================================================
// MAIN
// =======================================================
int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    int threadsPerBlock = BLOCK_SIZE; // Χρήση της σταθεράς μας
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "--- Starting CUDA Memory Models Experiment (Static Shared) ---" << std::endl;
    std::cout << "N = " << N << ", Threads Per Block = " << threadsPerBlock << ", Total Blocks = " << blocks << "\n" << std::endl;

    // -------------------------------------------------------
    // ΠΕΙΡΑΜΑ 1 & 2: REGISTERS & GLOBAL MEMORY (Χρειάζονται μέγεθος N)
    // -------------------------------------------------------
    double *h_out_N = new double[N];
    double *d_out_N;
    cudaMalloc(&d_out_N, N * sizeof(double)); 

    // --- Τρέξιμο του Kernel 1 (Registers) ---
    auto start = std::chrono::high_resolution_clock::now();
    kernel_registers<<<blocks, threadsPerBlock>>>(a, h, d_out_N);
    cudaDeviceSynchronize();

    
    cudaMemcpy(h_out_N, d_out_N, N * sizeof(double), cudaMemcpyDeviceToHost);
    
    double sum_reg = 0.0;
    for (int i = 0; i < N; i++) sum_reg += h_out_N[i];
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_reg = end - start;
    std::cout << "1. REGISTER CASE    -> Integral = " << sum_reg << " | Time = " << elapsed_reg.count() << " sec" << std::endl;

    // --- Τρέξιμο του Kernel 2 (Global Memory) ---
    start = std::chrono::high_resolution_clock::now();
    kernel_global<<<blocks, threadsPerBlock>>>(a, h, d_out_N);
    cudaDeviceSynchronize();
    
    
    cudaMemcpy(h_out_N, d_out_N, N * sizeof(double), cudaMemcpyDeviceToHost);
    
    double sum_glob = 0.0;
    for (int i = 0; i < N; i++) sum_glob += h_out_N[i];
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_glob = end - start;
    std::cout << "2. GLOBAL CASE      -> Integral = " << sum_glob << " | Time = " << elapsed_glob.count() << " sec" << std::endl;

    // Αποδέσμευση της μνήμης μεγέθους N
    cudaFree(d_out_N);
    delete[] h_out_N;

    // -------------------------------------------------------
    // ΠΕΙΡΑΜΑ 3: SHARED MEMORY REDUCTION (Χρειάζεται μέγεθος blocks)
    // -------------------------------------------------------
    double *h_out_blocks = new double[blocks];
    double *d_out_blocks;
    cudaMalloc(&d_out_blocks, blocks * sizeof(double)); 

    start = std::chrono::high_resolution_clock::now();
    // Καθαρή κλήση χωρίς το 3ο όρισμα, αφού η shared μνήμη ορίστηκε στατικά
    kernel_shared<<<blocks, threadsPerBlock>>>(a, h, d_out_blocks, N);
    cudaDeviceSynchronize();
  

    cudaMemcpy(h_out_blocks, d_out_blocks, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_shared = 0.0;
    for (int i = 0; i < blocks; i++) sum_shared += h_out_blocks[i];
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_shared = end - start;
    std::cout << "3. SHARED (REDUCE)  -> Integral = " << sum_shared << " | Time = " << elapsed_shared.count() << " sec" << std::endl;

    // Αποδέσμευση της μνήμης μεγέθους blocks
    cudaFree(d_out_blocks);
    delete[] h_out_blocks;

    return 0;
}