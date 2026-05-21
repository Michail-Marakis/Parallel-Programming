#include <iostream>
#include <cmath>
#include <chrono>
#include <CL/cl.h>

#define N 100000
#define BLOCK_SIZE 256

const char* kernelSource = R"(
double f(double x) {
    return x * x;
}

// =======================================================
// 1. REGISTER CASE (LOCAL COMPUTATION WITH REGISTERS)
// =======================================================
__kernel void kernel_registers(double a, double h, __global double* out, int n_elements) {
    int tid = get_global_id(0);

    if (tid < n_elements) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        double val = (f(x1) + f(x2)) * h / 2.0;
        out[tid] = val;
    }
}

// =======================================================
// 2. GLOBAL CASE (DIRECT GLOBAL MEMORY ACCESS)
// =======================================================
__kernel void kernel_global(double a, double h, __global double* out, int n_elements) {
    int tid = get_global_id(0);

    if (tid < n_elements) {
        out[tid] = (f(a + tid * h) + f(a + (tid + 1) * h)) * h / 2.0;
    }
}

// =======================================================
// 3. SHARED CASE (COMPUTE & BLOCK-REDUCTION IN ONE GO)
// =======================================================
__kernel void kernel_shared_init(double a, double h, __global double* output, int n_elements, __local double* cache) {
    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    double val = 0.0;
    if (tid < n_elements) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        val = (f(x1) + f(x2)) * h / 2.0;
    }

    cache[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = local_size / 2; s > 0; s /= 2) {
        if (lid < s) {
            cache[lid] += cache[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[group_id] = cache[0];
    }
}

// =======================================================
// COMMON REDUCTION KERNEL (FOR PIPELINE)
// =======================================================
__kernel void reduce(__global double* input, __global double* output, int n, __local double* cache) {
    int tid = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);

    double val = (tid < n) ? input[tid] : 0.0;
    cache[local_id] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = local_size / 2; s > 0; s /= 2) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output[group_id] = cache[0];
    }
}
)";

// Βοηθητική συνάρτηση για την εκτέλεση του Full GPU Reduction με εκτυπώσεις βημάτων
double run_gpu_reduction(cl_command_queue q, cl_kernel k_reduce, cl_mem d_in, cl_mem d_tmp, int starting_n, size_t local_size) {
    int n = starting_n;
    cl_mem in_buf = d_in;
    cl_mem out_buf = d_tmp;
    int step = 1;

    while (n > 1) {
        size_t current_local = local_size;
        size_t current_global = ((n + current_local - 1) / current_local) * current_local;
        size_t num_groups = current_global / current_local;

        // Εκτύπωση της προόδου του reduction στην GPU
        std::cout << "   [Reduction Step " << step << "] Reducing " << n 
                  << " elements into " << num_groups << " blocks...\n";

        clSetKernelArg(k_reduce, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(k_reduce, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(k_reduce, 2, sizeof(int), &n);
        clSetKernelArg(k_reduce, 3, current_local * sizeof(double), NULL);

        clEnqueueNDRangeKernel(q, k_reduce, 1, NULL, &current_global, &current_local, 0, NULL, NULL);
        clFinish(q); // Host-side Global Synchronization

        n = num_groups;
        cl_mem tmp = in_buf;
        in_buf = out_buf;
        out_buf = tmp;
        step++;
    }

    double final_result;
    clEnqueueReadBuffer(q, in_buf, CL_TRUE, 0, sizeof(double), &final_result, 0, NULL, NULL);
    return final_result;
}

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, NULL);

    cl_program p = clCreateProgramWithSource(ctx, 1, &kernelSource, NULL, NULL);
    clBuildProgram(p, 1, &device, NULL, NULL, NULL);

    cl_kernel k_reg = clCreateKernel(p, "kernel_registers", NULL);
    cl_kernel k_glob = clCreateKernel(p, "kernel_global", NULL);
    cl_kernel k_shr_init = clCreateKernel(p, "kernel_shared_init", NULL);
    cl_kernel k_reduce = clCreateKernel(p, "reduce", NULL);

    size_t local = BLOCK_SIZE;
    size_t global = ((N + local - 1) / local) * local;
    size_t blocks = global / local;
    int n_elements = N;

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(double), NULL, NULL);
    cl_mem d_tmp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(double), NULL, NULL);
    cl_mem d_blocks = clCreateBuffer(ctx, CL_MEM_READ_WRITE, blocks * sizeof(double), NULL, NULL);

    std::cout << "--- OPENCL MEMORY MODELS (FULL GPU REDUCTION PIPELINE) ---\n";
    std::cout << "N = " << N << " | Block Size = " << BLOCK_SIZE << "\n\n";

    // =======================================================
    // 1. CASE: REGISTERS
    // =======================================================
    std::cout << "Running REGISTER CASE...\n";
    clSetKernelArg(k_reg, 0, sizeof(double), &a);
    clSetKernelArg(k_reg, 1, sizeof(double), &h);
    clSetKernelArg(k_reg, 2, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_reg, 3, sizeof(int), &n_elements);

    auto t1 = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(q, k_reg, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(q);
    
    double res_reg = run_gpu_reduction(q, k_reduce, d_in, d_tmp, N, local);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    std::cout << ">> RESULT = " << res_reg 
              << " | Time = " << std::chrono::duration<double>(t2 - t1).count() << " sec\n\n";

    // =======================================================
    // 2. CASE: GLOBAL MEMORY
    // =======================================================
    std::cout << "Running GLOBAL CASE...\n";
    clSetKernelArg(k_glob, 0, sizeof(double), &a);
    clSetKernelArg(k_glob, 1, sizeof(double), &h);
    clSetKernelArg(k_glob, 2, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_glob, 3, sizeof(int), &n_elements);

    auto t3 = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(q, k_glob, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(q);
    
    double res_glob = run_gpu_reduction(q, k_reduce, d_in, d_tmp, N, local);
    auto t4 = std::chrono::high_resolution_clock::now();
    
    std::cout << ">> RESULT = " << res_glob 
              << " | Time = " << std::chrono::duration<double>(t4 - t3).count() << " sec\n\n";

    // =======================================================
    // 3. CASE: SHARED MEMORY
    // =======================================================
    std::cout << "Running SHARED CASE...\n";
    clSetKernelArg(k_shr_init, 0, sizeof(double), &a);
    clSetKernelArg(k_shr_init, 1, sizeof(double), &h);
    clSetKernelArg(k_shr_init, 2, sizeof(cl_mem), &d_blocks);
    clSetKernelArg(k_shr_init, 3, sizeof(int), &n_elements);
    clSetKernelArg(k_shr_init, 4, local * sizeof(double), NULL);

    auto t5 = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(q, k_shr_init, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(q);
    
    // Ξεκινάει απευθείας από τον αριθμό των blocks, γλιτώνοντας βήματα!
    double res_shr = run_gpu_reduction(q, k_reduce, d_blocks, d_tmp, blocks, local);
    auto t6 = std::chrono::high_resolution_clock::now();
    
    std::cout << ">> RESULT = " << res_shr 
              << " | Time = " << std::chrono::duration<double>(t6 - t5).count() << " sec\n\n";

    // Καθαρισμός
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_tmp);
    clReleaseMemObject(d_blocks);
    clReleaseKernel(k_reg);
    clReleaseKernel(k_glob);
    clReleaseKernel(k_shr_init);
    clReleaseKernel(k_reduce);
    clReleaseProgram(p);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}