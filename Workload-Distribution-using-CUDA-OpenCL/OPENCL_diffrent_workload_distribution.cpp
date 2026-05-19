#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <CL/cl.h>

#define N 100000

// -------------------- OPENCL KERNEL SOURCE CODE --------------------
// Ο kernel ορίζεται ως string διότι το OpenCL τον κάνει compile στο runtime
const char* kernelSource = R"(
double f(double x) {
    return x * x;
}

__kernel void integrate(double a, double h, __global double* partial_sum, int mode) {
    //Αντίστοιχο του CUDA: blockIdx.x * blockDim.x + threadIdx.x
    int tid = get_global_id(0); 
    
    //Αντίστοιχο του CUDA: blockDim.x * gridDim.x
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
    // INITIALIZE OPENCL ENVIRONMENT
    // ------------------------------------------------------------------
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_devices, num_platforms;

    // Πλατφόρμα & Συσκευή (Επιλογή GPU βάσει των slides)
    clGetPlatformIDs(1, &platform_id, &num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    // Δημιουργία Context και Command Queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, NULL);

    // Δημιουργία και Compile του Program από το String
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Δημιουργία του Kernel αντικειμένου
    cl_kernel kernel = clCreateKernel(program, "integrate", NULL);

    // Δέσμευση Μνήμης στη GPU (Device Buffer)
    cl_mem d_partial = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(double), NULL, NULL);
    
    // Μνήμη Host (CPU)
    double* h_partial = new double[N];

    // Ρύθμιση Work-group Sizes (Αντίστοιχο των threadsPerBlock)
    size_t localWorkSize = 256; // Block Size

    // ------------------------------------------------------------------
    // EXECUTION LOOP FOR MODES
    // ------------------------------------------------------------------
    for (int mode = 0; mode <= 1; mode++) {
        
        size_t globalWorkSize; // Συνολικά Work-items (Grid Size)
        if (mode == 0) {
            // mode = 0: Στρογγυλοποίηση στο κοντινότερο πολλαπλάσιο του 256
            globalWorkSize = ((N + localWorkSize - 1) / localWorkSize) * localWorkSize;
        } else {
            // mode = 1: Σταθερό πλήθος από 1024 blocks * 256 threads = 262144 work-items
            globalWorkSize = 1024 * localWorkSize;
        }

        // Ορισμός των ορισμάτων του Kernel
        clSetKernelArg(kernel, 0, sizeof(double), &a);
        clSetKernelArg(kernel, 1, sizeof(double), &h);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_partial);
        clSetKernelArg(kernel, 3, sizeof(int), &mode);

        // --- ΕΝΑΡΞΗ ΧΡΟΝΟΜΕΤΡΗΣΗΣ ---
        auto start = std::chrono::high_resolution_clock::now();

        // Εκκίνηση του Kernel (NDRange)
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

        // Σύγχρονη αναμονή για την ολοκλήρωση της GPU (Αντίστοιχο του cudaDeviceSynchronize)
        clFinish(queue);

        // --- ΛΗΞΗ ΧΡΟΝΟΜΕΤΡΗΣΗΣ (Σωστό σημείο, πάνω από το Copy) ---
        auto end = std::chrono::high_resolution_clock::now();

        // Μεταφορά δεδομένων στη CPU (Αντίστοιχο του cudaMemcpy)
        clEnqueueReadBuffer(queue, d_partial, CL_TRUE, 0, N * sizeof(double), h_partial, 0, NULL, NULL);

        // Σειριακό Άθροισμα στη CPU
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += h_partial[i];
        }

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\nOPENCL MODE " << mode << " (Global Size: " << globalWorkSize << ")" << std::endl;
        std::cout << "  Integral = " << sum << std::endl;
        std::cout << "  GPU Time = " << elapsed.count() << " sec" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
    }

    // Καθαρισμός Μνήμης OpenCL (Αντίστοιχο των cudaFree)
    clReleaseMemObject(d_partial);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    delete[] h_partial;
    return 0;
}