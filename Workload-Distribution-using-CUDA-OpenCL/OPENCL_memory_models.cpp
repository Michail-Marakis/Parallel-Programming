#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <CL/cl.h>

#define N 100000 
#define BLOCK_SIZE 256 

const char* kernelSource = R"(
float f(float x) {
    return x * x;
}

__kernel void kernel_registers(float a, float h, __global float* out) {
    int tid = get_global_id(0);
    if (tid < 100000) {
        float x1 = a + tid * h;
        float x2 = a + (tid + 1) * h;
        float val = (f(x1) + f(x2)) * h / 2.0f; // Private Memory
        out[tid] = val;
    }
}

__kernel void kernel_global(float a, float h, __global float* out) {
    int tid = get_global_id(0);
    if (tid < 100000) {
        float x1 = a + tid * h;
        float x2 = a + (tid + 1) * h;
        out[tid] = (f(x1) + f(x2)) * h / 2.0f; // Απευθείας Global Memory
    }
}

__kernel void kernel_shared(float a, float h, __global float* partial, int n_elements) {
    __local float cache[128]; // Local Memory (Διαφάνεια 53)
    int tid = get_local_id(0); 
    int i = get_global_id(0);  

    float value = 0.0f;
    if (i < n_elements) {
        float x1 = a + i * h;
        float x2 = a + (i + 1) * h;
        value = (f(x1) + f(x2)) * h * 0.5f;
    }

    cache[tid] = value;
    barrier(CLK_LOCAL_MEM_FENCE); // Συγχρονισμός 


    for (int step = get_local_size(0) / 2; step > 0; step /= 2) {
        if (tid < step) {
            cache[tid] += cache[tid + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
    }

    if (tid == 0) {
        partial[get_group_id(0)] = cache[0];
    }
}
)";

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during: " << operation << " (Code: " << err << ")" << std::endl;
        exit(1);
    }
}

int main() {
    float a = 0.0f, b = 10.0f;
    float h = (b - a) / N;

    size_t localWorkSize = BLOCK_SIZE; 
    size_t globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;
    int total_blocks = globalWorkSize / localWorkSize;

    std::cout << "--- Intel GPU OpenCL Memory Models Experiment ---" << std::endl;
    std::cout << "N = " << N << ", Local Size = " << localWorkSize << ", Total Groups = " << total_blocks << "\n" << std::endl;

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_int err;

    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // --- ΠΕΙΡΑΜΑ 1 & 2 ---
    cl_mem d_out_N = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);
    float* h_out_N = new float[N];

    // Kernel 1
    cl_kernel k_regs = clCreateKernel(program, "kernel_registers", &err);
    clSetKernelArg(k_regs, 0, sizeof(float), &a);
    clSetKernelArg(k_regs, 1, sizeof(float), &h);
    clSetKernelArg(k_regs, 2, sizeof(cl_mem), &d_out_N);

    auto start = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, k_regs, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    checkError(err, "Launch Kernel 1");
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, d_out_N, CL_TRUE, 0, N * sizeof(float), h_out_N, 0, NULL, NULL);
    float sum_reg = 0.0f;
    for (int i = 0; i < N; i++) sum_reg += h_out_N[i];
    std::chrono::duration<double> elapsed_reg = end - start;
    std::cout << "1. REGISTER CASE    -> Integral = " << sum_reg << " | Time = " << elapsed_reg.count() << " sec" << std::endl;

    // Kernel 2
    cl_kernel k_glob = clCreateKernel(program, "kernel_global", &err);
    clSetKernelArg(k_glob, 0, sizeof(float), &a);
    clSetKernelArg(k_glob, 1, sizeof(float), &h);
    clSetKernelArg(k_glob, 2, sizeof(cl_mem), &d_out_N);

    start = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, k_glob, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    checkError(err, "Launch Kernel 2");
    clFinish(queue);
    end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, d_out_N, CL_TRUE, 0, N * sizeof(float), h_out_N, 0, NULL, NULL);
    float sum_glob = 0.0f;
    for (int i = 0; i < N; i++) sum_glob += h_out_N[i];
    std::chrono::duration<double> elapsed_glob = end - start;
    std::cout << "2. GLOBAL CASE      -> Integral = " << sum_glob << " | Time = " << elapsed_glob.count() << " sec" << std::endl;

    clReleaseMemObject(d_out_N);
    delete[] h_out_N;

    // --- ΠΕΙΡΑΜΑ 3 ---
    cl_mem d_out_blocks = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_blocks * sizeof(float), NULL, &err);
    float* h_out_blocks = new float[total_blocks];
    int n_elements = N;

    cl_kernel k_shared = clCreateKernel(program, "kernel_shared", &err);
    clSetKernelArg(k_shared, 0, sizeof(float), &a);
    clSetKernelArg(k_shared, 1, sizeof(float), &h);
    clSetKernelArg(k_shared, 2, sizeof(cl_mem), &d_out_blocks);
    clSetKernelArg(k_shared, 3, sizeof(int), &n_elements);

    start = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, k_shared, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    checkError(err, "Launch Kernel 3");
    clFinish(queue);
    end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, d_out_blocks, CL_TRUE, 0, total_blocks * sizeof(float), h_out_blocks, 0, NULL, NULL);
    float sum_shared = 0.0f;
    for (int i = 0; i < total_blocks; i++) sum_shared += h_out_blocks[i];
    std::chrono::duration<double> elapsed_shared = end - start;
    std::cout << "3. LOCAL (REDUCE)   -> Integral = " << sum_shared << " | Time = " << elapsed_shared.count() << " sec" << std::endl;

    clReleaseMemObject(d_out_blocks);
    delete[] h_out_blocks;
    clReleaseKernel(k_regs); clReleaseKernel(k_glob); clReleaseKernel(k_shared);
    clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}