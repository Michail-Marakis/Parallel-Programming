#include <iostream>
#include <cmath>
#include <chrono>
#include <CL/cl.h>

#define N 100000000
#define BLOCK_SIZE 128

const char* kernelSource = R"(
double f(double x) {
    return x * x;
}

// =======================================================
// 1. REGISTER CASE
// =======================================================
__kernel void kernel_registers(double a, double h, __global double* output, int use_barrier, int n_elements) {
    __local double cache[BLOCK_SIZE];
    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // Υπολογισμός με τοπικές μεταβλητές (Registers)
    double val = 0.0;
    if (tid < n_elements) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        val = (f(x1) + f(x2)) * h / 2.0;
    }

    cache[lid] = val;

    if (use_barrier) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int s = local_size / 2; s > 0; s /= 2) {
        if (lid < s) {
            cache[lid] += cache[lid + s];
        }
        if (use_barrier) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (lid == 0) {
        output[get_group_id(0)] = cache[0];
    }
}

// =======================================================
// 2. GLOBAL CASE
// =======================================================
__kernel void kernel_global(double a, double h, __global double* output, int use_barrier, int n_elements) {
    __local double cache[BLOCK_SIZE];
    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // Απευθείας ανάγνωση και εγγραφή από/προς τη Global μνήμη χωρίς ενδιάμεσο register
    if (tid < n_elements) {
        cache[lid] = (f(a + tid * h) + f(a + (tid + 1) * h)) * h / 2.0;
    } else {
        cache[lid] = 0.0;
    }

    if (use_barrier) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int s = local_size / 2; s > 0; s /= 2) {
        if (lid < s) {
            cache[lid] += cache[lid + s];
        }
        if (use_barrier) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (lid == 0) {
        output[get_group_id(0)] = cache[0];
    }
}

// =======================================================
// 3. SHARED CASE
// =======================================================
__kernel void kernel_shared(double a, double h, __global double* output, int use_barrier, int n_elements, __local double* cache) {
    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    double val = 0.0;
    if (tid < n_elements) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        val = (f(x1) + f(x2)) * h / 2.0;
    }

    // Ρητή χρήση της __local μνήμης για την αποθήκευση πριν από οποιαδήποτε άλλη πράξη
    cache[lid] = val;

    if (use_barrier) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int s = local_size / 2; s > 0; s /= 2) {
        if (lid < s) {
            cache[lid] += cache[lid + s];
        }
        if (use_barrier) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (lid == 0) {
        output[get_group_id(0)] = cache[0];
    }
}
)";

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    cl_platform_id platform_id;
    cl_device_id device_id;
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel k_reg = clCreateKernel(program, "kernel_registers", NULL);
    cl_kernel k_glob = clCreateKernel(program, "kernel_global", NULL);
    cl_kernel k_shr = clCreateKernel(program, "kernel_shared", NULL);

    size_t localWorkSize = BLOCK_SIZE;
    size_t globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;
    size_t numGroups = globalWorkSize / localWorkSize;
    int n_local = N;

    cl_mem d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * sizeof(double), NULL, NULL);
    double* h_partial = new double[numGroups];

    std::cout << "--- OPENCL MEMORY MODELS & BARRIER TEST (SIMPLE FLOW) ---\n";

    for (int use_barrier = 0; use_barrier <= 1; use_barrier++) {
        std::cout << "\n==================================";
        std::cout << "\n TESTING WITH BARRIER = " << use_barrier << "\n==================================\n";

        // 1. REGISTER CASE
        clSetKernelArg(k_reg, 0, sizeof(double), &a);
        clSetKernelArg(k_reg, 1, sizeof(double), &h);
        clSetKernelArg(k_reg, 2, sizeof(cl_mem), &d_output);
        clSetKernelArg(k_reg, 3, sizeof(int), &use_barrier);
        clSetKernelArg(k_reg, 4, sizeof(int), &n_local);

        auto start1 = std::chrono::high_resolution_clock::now();
        clEnqueueNDRangeKernel(queue, k_reg, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        clFinish(queue);
        auto end1 = std::chrono::high_resolution_clock::now();

        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, numGroups * sizeof(double), h_partial, 0, NULL, NULL);
        double sum1 = 0.0;
        for (size_t i = 0; i < numGroups; i++) sum1 += h_partial[i];
        std::cout << "1. REGISTER -> Integral = " << sum1 << " | Time = " << std::chrono::duration<double>(end1 - start1).count() << " sec\n";

        // 2. GLOBAL CASE
        clSetKernelArg(k_glob, 0, sizeof(double), &a);
        clSetKernelArg(k_glob, 1, sizeof(double), &h);
        clSetKernelArg(k_glob, 2, sizeof(cl_mem), &d_output);
        clSetKernelArg(k_glob, 3, sizeof(int), &use_barrier);
        clSetKernelArg(k_glob, 4, sizeof(int), &n_local);

        auto start2 = std::chrono::high_resolution_clock::now();
        clEnqueueNDRangeKernel(queue, k_glob, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        clFinish(queue);
        auto end2 = std::chrono::high_resolution_clock::now();

        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, numGroups * sizeof(double), h_partial, 0, NULL, NULL);
        double sum2 = 0.0;
        for (size_t i = 0; i < numGroups; i++) sum2 += h_partial[i];
        std::cout << "2. GLOBAL   -> Integral = " << sum2 << " | Time = " << std::chrono::duration<double>(end2 - start2).count() << " sec\n";

        // 3. SHARED CASE
        clSetKernelArg(k_shr, 0, sizeof(double), &a);
        clSetKernelArg(k_shr, 1, sizeof(double), &h);
        clSetKernelArg(k_shr, 2, sizeof(cl_mem), &d_output);
        clSetKernelArg(k_shr, 3, sizeof(int), &use_barrier);
        clSetKernelArg(k_shr, 4, sizeof(int), &n_local);
        clSetKernelArg(k_shr, 5, localWorkSize * sizeof(double), NULL); // Δυναμικό allocation τοπικής μνήμης

        auto start3 = std::chrono::high_resolution_clock::now();
        clEnqueueNDRangeKernel(queue, k_shr, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        clFinish(queue);
        auto end3 = std::chrono::high_resolution_clock::now();

        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, numGroups * sizeof(double), h_partial, 0, NULL, NULL);
        double sum3 = 0.0;
        for (size_t i = 0; i < numGroups; i++) sum3 += h_partial[i];
        std::cout << "3. SHARED   -> Integral = " << sum3 << " | Time = " << std::chrono::duration<double>(end3 - start3).count() << " sec\n";
    }

    // Cleanup
    delete[] h_partial;
    clReleaseMemObject(d_output);
    clReleaseKernel(k_reg);
    clReleaseKernel(k_glob);
    clReleaseKernel(k_shr);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}