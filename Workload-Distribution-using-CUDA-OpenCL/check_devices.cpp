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

__kernel void integrate(double a, double h, __global double* partial_sum, int mode) {
    int tid = get_global_id(0); 
    int stride = get_global_size(0); 

    // -------- MODE 0: 1 element / thread ----------------
    if (mode == 0) {
        if (tid < 100000) { 
            double x1 = a + tid * h;
            double x2 = a + (tid + 1) * h;
            partial_sum[tid] = (f(x1) + f(x2)) * h / 2.0;
        }
    }
    // ------------ MODE 1: many elements / thread ----------------
    else {
        for (int i = tid; i < 100000; i += stride) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            partial_sum[i] = (f(x1) + f(x2)) * h / 2.0;
        }
    }
}
)";

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;

    // ------------------------------------------------------------------
    // INITIALIZE OPENCL ENVIRONMENT FOR CPU
    // ------------------------------------------------------------------
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);

    cl_device_id device_id = NULL;
    cl_uint num_devices = 0;
    cl_int err;

    // Σκανάρισμα όλων των πλατφορμών για εύρεση CPU
    for (cl_uint i = 0; i < num_platforms; i++) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device_id, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            char platform_name[128];
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
            std::cout << "[OpenCL] Using CPU via platform: " << platform_name << std::endl;
            break;
        }
    }

    // Fallback: Αν δεν βρεθεί CPU runtime driver, αναγκαστική χρήση της GPU
    if (num_devices == 0) {
        std::cout << "[Warning] OpenCL CPU Device not found (missing drivers). Falling back to GPU..." << std::endl;
        clGetPlatformIDs(1, &platforms[0], NULL);
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    }

    // Εκτύπωση του ονόματος της συσκευής που επιλέχθηκε τελικά
    char device_name[128];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::cout << "[OpenCL] Device Selected: " << device_name << "\n" << std::endl;

    // Δημιουργία Context και Command Queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Δημιουργία και Compile του Program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Δημιουργία του Kernel
    cl_kernel kernel = clCreateKernel(program, "integrate", &err);

    // Δέσμευση Μνήμης
    cl_mem d_partial = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(double), NULL, &err);
    double* h_partial = new double[N];

    size_t localWorkSize = 256; 

    // ------------------------------------------------------------------
    // EXECUTION LOOP FOR MODES
    // ------------------------------------------------------------------
    for (int mode = 0; mode <= 1; mode++) {
        
        size_t globalWorkSize; 
        if (mode == 0) {
            globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;
        } else {
            globalWorkSize = 1024 * localWorkSize;
        }

        clSetKernelArg(kernel, 0, sizeof(double), &a);
        clSetKernelArg(kernel, 1, sizeof(double), &h);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_partial);
        clSetKernelArg(kernel, 3, sizeof(int), &mode);

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
        std::cout << "OPENCL MODE " << mode << " (Global Size: " << globalWorkSize << ")" << std::endl;
        std::cout << "  Integral = " << sum << std::endl;
        std::cout << "  Device Time = " << elapsed.count() << " sec" << std::endl;
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