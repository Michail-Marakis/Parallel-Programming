#include <iostream>
#include <cmath>
#include <chrono>
#include <CL/cl.h>

#define N 100000000

const char* kernelSource = R"(
double f(double x) {
    return x * x;
}

__kernel void integrate_reduce(double a, double h,
                               __global double* output,
                               int N,
                               int mode) {

    int tid = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int stride = get_global_size(0);

    __local double cache[256];

    double val = 0.0;

    // -------- MODE 0: 1 element / thread --------
    if (mode == 0) {
        if (tid < N) {
            double x1 = a + tid * h;
            double x2 = a + (tid + 1) * h;
            val = (f(x1) + f(x2)) * h / 2.0;
        }
    }
    // -------- MODE 1: many elements / thread --------
    else {
        for (int i = tid; i < N; i += stride) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            val += (f(x1) + f(x2)) * h / 2.0;
        }
    }

    cache[local_id] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // -------- PARALLEL REDUCTION --------
    for (int s = local_size / 2; s > 0; s /= 2) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output[get_group_id(0)] = cache[0];
    }
}

__kernel void reduce(__global double* input,
                     __global double* output,
                     int N) {

    int tid = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    __local double cache[256];

    double val = 0.0;
    if (tid < N) {
        val = input[tid];
    }

    cache[local_id] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = local_size / 2; s > 0; s /= 2) {
        if (local_id < s) {
            cache[local_id] += cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output[get_group_id(0)] = cache[0];
    }
}
)";

int main() {

    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    cl_platform_id platform;
    cl_device_id device;
    cl_uint num;

    clGetPlatformIDs(1, &platform, &num);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel k_integrate = clCreateKernel(program, "integrate_reduce", NULL);
    cl_kernel k_reduce = clCreateKernel(program, "reduce", NULL);

    size_t localWorkSize = 256;

    for (int mode = 0; mode <= 1; mode++) {

        size_t globalWorkSize;

        if (mode == 0) {
            globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;
        } else {
            globalWorkSize = 1024 * localWorkSize;
        }

        size_t firstGroups = globalWorkSize / localWorkSize;

        cl_mem d_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         firstGroups * sizeof(double), NULL, NULL);

        // -------- FIRST PASS --------
        clSetKernelArg(k_integrate, 0, sizeof(double), &a);
        clSetKernelArg(k_integrate, 1, sizeof(double), &h);
        clSetKernelArg(k_integrate, 2, sizeof(cl_mem), &d_output);
        int N_val = N;
        clSetKernelArg(k_integrate, 3, sizeof(int), &N_val);
        clSetKernelArg(k_integrate, 4, sizeof(int), &mode);

        auto start = std::chrono::high_resolution_clock::now();

        clEnqueueNDRangeKernel(queue, k_integrate, 1, NULL,
                               &globalWorkSize, &localWorkSize, 0, NULL, NULL);

        clFinish(queue);

        // -------- MULTI PASS REDUCTION --------
        size_t currentSize = firstGroups;
        cl_mem d_in = d_output;
        cl_mem d_out;

        while (currentSize > 1) {

            size_t newGlobal = ((currentSize + localWorkSize - 1) / localWorkSize) * localWorkSize;
            size_t numGroups = newGlobal / localWorkSize;

            d_out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   numGroups * sizeof(double), NULL, NULL);

            clSetKernelArg(k_reduce, 0, sizeof(cl_mem), &d_in);
            clSetKernelArg(k_reduce, 1, sizeof(cl_mem), &d_out);
            clSetKernelArg(k_reduce, 2, sizeof(int), &currentSize);

            clEnqueueNDRangeKernel(queue, k_reduce, 1, NULL,
                                   &newGlobal, &localWorkSize, 0, NULL, NULL);

            clFinish(queue);

            clReleaseMemObject(d_in);
            d_in = d_out;
            currentSize = numGroups;
        }

        auto end = std::chrono::high_resolution_clock::now();

        double result;
        clEnqueueReadBuffer(queue, d_in, CL_TRUE, 0,
                            sizeof(double), &result, 0, NULL, NULL);

        std::chrono::duration<double> elapsed = end - start;

        std::cout << "\nMODE " << mode << std::endl;
        std::cout << "Integral = " << result << std::endl;
        std::cout << "GPU Time = " << elapsed.count() << " sec" << std::endl;
        std::cout << "----------------------------------" << std::endl;

        clReleaseMemObject(d_in);
    }

    clReleaseKernel(k_integrate);
    clReleaseKernel(k_reduce);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}