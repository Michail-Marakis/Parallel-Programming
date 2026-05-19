#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <CL/cl.h>

#define N 100000

// -------------------- OPENCL KERNEL SOURCE CODE --------------------
const char* kernelSource = R"(
double f(double x) {
    return x * x;
}

__kernel void integrate(double a, double h, __global double* partial_sum) {
    // Αντίστοιχο του blockIdx.x * blockDim.x + threadIdx.x
    int tid = get_global_id(0); 

    if (tid < 100000) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        partial_sum[tid] = (f(x1) + f(x2)) * h / 2.0;
    }
}
)";

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    // ------------------------------------------------------------------
    // INITIALIZE OPENCL ENVIRONMENT
    // ------------------------------------------------------------------
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_devices, num_platforms;

    clGetPlatformIDs(1, &platform_id, &num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "integrate", NULL);

    // Δέσμευση Μνήμης στη GPU (Device Buffer)
    cl_mem d_partial = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(double), NULL, NULL);
    
    // Μνήμη Host (CPU)
    double* h_partial = new double[N];

    // Τα διαφορετικά μεγέθη ομάδων (Block Sizes / Local Work Sizes) που ζητάει η εκφώνηση
    int blockSizes[] = {32, 64, 128, 256, 512};

    std::cout << "--- Starting OpenCL Block Size Experiment ---" << std::endl;

    for (int bsize : blockSizes) {

        // Στο OpenCL το "threadsPerBlock" ονομάζεται localWorkSize
        size_t localWorkSize = bsize; 

        // Το συνολικό μέγεθος του Grid (globalWorkSize) πρέπει να είναι πολλαπλάσιο του localWorkSize
        size_t globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;

        // Ορισμός των ορισμάτων του Kernel
        clSetKernelArg(kernel, 0, sizeof(double), &a);
        clSetKernelArg(kernel, 1, sizeof(double), &h);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_partial);

        // --- ΕΝΑΡΞΗ ΧΡΟΝΟΜΕΤΡΗΣΗΣ ---
        auto start = std::chrono::high_resolution_clock::now();

        // Εκκίνηση του Kernel (NDRange)
        cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cout << "OpenCL Launch Error Code: " << err << " for Block Size " << bsize << std::endl;
        }

        // Σύγχρονη αναμονή για την ολοκλήρωση της GPU (Αντίστοιχο του cudaDeviceSynchronize)
        clFinish(queue);

        // --- ΛΗΞΗ ΧΡΟΝΟΜΕΤΡΗΣΗΣ (10/10 - Πάνω από το Copy!) ---
        auto end = std::chrono::high_resolution_clock::now();

        // Μεταφορά δεδομένων στη CPU (Αντίστοιχο του cudaMemcpy)
        clEnqueueReadBuffer(queue, d_partial, CL_TRUE, 0, N * sizeof(double), h_partial, 0, NULL, NULL);

        // Σειριακό Άθροισμα στη CPU
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += h_partial[i];
        }

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "\nBLOCK SIZE (Local Size) = " << bsize << std::endl;
        std::cout << "  Global Size = " << globalWorkSize << std::endl;
        std::cout << "  Integral    = " << sum << std::endl;
        std::cout << "  GPU Time    = " << elapsed.count() << " sec" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }

    // Καθαρισμός Μνήμης
    clReleaseMemObject(d_partial);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    delete[] h_partial;
    return 0;
}