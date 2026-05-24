#include <iostream>
#include <cmath>
#include <chrono>
#include <CL/cl.h>

#define N 100000000
#define BLOCK_SIZE 128 // local size

const char* kernelSource = R"(

double f(double x) {
    return x * x;
}


//REGISTER CASE

__kernel void kernel_registers(double a, double h, __global double* output, int use_barrier, int n_elements) {

    __local double cache[BLOCK_SIZE]; //local memory

    int tid = get_global_id(0); //global id
    int lid = get_local_id(0);  //local id
    int local_size = get_local_size(0);

    //υπολογισμός σε register
    double val = 0.0;

    if (tid < n_elements) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;

        val = (f(x1) + f(x2)) * h / 2.0;
    }

    cache[lid] = val; //write σε local

    if (use_barrier) {
        barrier(CLK_LOCAL_MEM_FENCE); //sync
    }

    //reduction
    for (int s = local_size / 2; s > 0; s /= 2) {

        if (lid < s) {
            cache[lid] += cache[lid + s];
        }

        if (use_barrier) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    //write block αποτέλεσμα
    if (lid == 0) {
        output[get_group_id(0)] = cache[0];
    }
}


//GLOBAL CASE
__kernel void kernel_global(double a, double h, __global double* output, int use_barrier, int n_elements) {

    __local double cache[BLOCK_SIZE];

    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // direct compute χωρίς register val
    if (tid < n_elements) {
        cache[lid] = (f(a + tid * h) + f(a + (tid + 1) * h)) * h / 2.0;
    } else {
        cache[lid] = 0.0;
    }

    if (use_barrier) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //reduction
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


//SHARED CASE
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

    cache[lid] = val; // write σε dynamic local

    if (use_barrier) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //reduction
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

    //πλατφόρμα + device
    cl_platform_id platform_id;
    cl_device_id device_id;

    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    //context + queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, NULL);

    //build program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    //kernels
    cl_kernel k_reg  = clCreateKernel(program, "kernel_registers", NULL);
    cl_kernel k_glob = clCreateKernel(program, "kernel_global", NULL);
    cl_kernel k_shr  = clCreateKernel(program, "kernel_shared", NULL);

    size_t localWorkSize = BLOCK_SIZE; //work-group size
    size_t globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize; 
    size_t numGroups = globalWorkSize / localWorkSize;

    int n_local = N;

    //output buffer
    cl_mem d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * sizeof(double), NULL, NULL);
    double* h_partial = new double[numGroups];

    std::cout << "--- OPENCL TEST ---\n";

    //barrier test
    for (int use_barrier = 0; use_barrier <= 1; use_barrier++) {

        std::cout << "\n=== BARRIER = " << use_barrier << " ===\n";

        //REGISTER
        clSetKernelArg(k_reg, 0, sizeof(double), &a);
        clSetKernelArg(k_reg, 1, sizeof(double), &h);
        clSetKernelArg(k_reg, 2, sizeof(cl_mem), &d_output);
        clSetKernelArg(k_reg, 3, sizeof(int), &use_barrier);
        clSetKernelArg(k_reg, 4, sizeof(int), &n_local);

        auto start1 = std::chrono::high_resolution_clock::now();

        clEnqueueNDRangeKernel(queue, k_reg, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        clFinish(queue);

        auto end1 = std::chrono::high_resolution_clock::now();

        //read + final sum
        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, numGroups * sizeof(double), h_partial, 0, NULL, NULL);

        double sum1 = 0.0;
        for (size_t i = 0; i < numGroups; i++) sum1 += h_partial[i];

        std::cout << "REGISTER -> " << sum1
                  << " | Time = " << std::chrono::duration<double>(end1 - start1).count() << "\n";

        //GLOBAL
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

        std::cout << "GLOBAL   -> " << sum2
                  << " | Time = " << std::chrono::duration<double>(end2 - start2).count() << "\n";

        //SHARED
        clSetKernelArg(k_shr, 0, sizeof(double), &a);
        clSetKernelArg(k_shr, 1, sizeof(double), &h);
        clSetKernelArg(k_shr, 2, sizeof(cl_mem), &d_output);
        clSetKernelArg(k_shr, 3, sizeof(int), &use_barrier);
        clSetKernelArg(k_shr, 4, sizeof(int), &n_local);
        clSetKernelArg(k_shr, 5, localWorkSize * sizeof(double), NULL); //dynamic local

        auto start3 = std::chrono::high_resolution_clock::now();

        clEnqueueNDRangeKernel(queue, k_shr, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        clFinish(queue);

        auto end3 = std::chrono::high_resolution_clock::now();

        clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, numGroups * sizeof(double), h_partial, 0, NULL, NULL);

        double sum3 = 0.0;
        for (size_t i = 0; i < numGroups; i++) sum3 += h_partial[i];

        std::cout << "SHARED   -> " << sum3
                  << " | Time = " << std::chrono::duration<double>(end3 - start3).count() << "\n";
    }

    //cleanup
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