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

__kernel void integrate_sync(double a, double h, __global double* partial_sum, int use_barrier) {
    
    // __local: Δημιουργία shared memory (Local Memory) μέσα στο Work-group
    __local double cache[256]; 

    int tid = get_global_id(0);   // Global thread ID
    int local_id = get_local_id(0); // Local thread ID inside the work-group

    double val = 0.0;

    if (tid < 100000) {
        double x1 = a + tid * h;
        double x2 = a + (tid + 1) * h;
        val = (f(x1) + f(x2)) * h / 2.0;
    }

    // Κάθε νήμα γράφει στην τοπική μνήμη (Shared/Local Memory)
    cache[local_id] = val;

    // Υλοποίηση του Barrier Συγχρονισμού
    if (use_barrier) {
        // Αντίστοιχο του __syncthreads(). 
        // Το CLK_LOCAL_MEM_FENCE επιβάλλει σε όλα τα work-items να περιμένουν
        // μέχρι να ολοκληρωθούν όλες οι εγγραφές στη Local Memory.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Ανάγνωση από την cache και εγγραφή στην Global Μνήμη
    if (tid < 100000) {
        partial_sum[tid] = cache[local_id];
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

    cl_kernel kernel = clCreateKernel(program, "integrate_sync", NULL);

    cl_mem d_partial = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(double), NULL, NULL);
    double* h_partial = new double[N];

    size_t localWorkSize = 256; // 256 threads per block/group
    size_t globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;

    for (int use_barrier = 0; use_barrier <= 1; use_barrier++) {

        clSetKernelArg(kernel, 0, sizeof(double), &a);
        clSetKernelArg(kernel, 1, sizeof(double), &h);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_partial);
        clSetKernelArg(kernel, 3, sizeof(int), &use_barrier);

        // --- ΕΝΑΡΞΗ ΧΡΟΝΟΜΕΤΡΗΣΗΣ ---
        auto start = std::chrono::high_resolution_clock::now();

        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        clFinish(queue);

        // --- ΛΗΞΗ ΧΡΟΝΟΜΕΤΡΗΣΗΣ ---
        auto end = std::chrono::high_resolution_clock::now();

        clEnqueueReadBuffer(queue, d_partial, CL_TRUE, 0, N * sizeof(double), h_partial, 0, NULL, NULL);

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += h_partial[i];
        }

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\nBARRIER = " << use_barrier << std::endl;
        std::cout << "  Integral = " << sum << std::endl;
        std::cout << "  Time     = " << elapsed.count() << " sec" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }

    clReleaseMemObject(d_partial);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    delete[] h_partial;

    return 0;
}