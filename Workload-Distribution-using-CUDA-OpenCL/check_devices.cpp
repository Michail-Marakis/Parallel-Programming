// #include <iostream>
// #include <vector>
// #include <CL/cl.h>

// int main() {
//     cl_uint num_platforms;
//     cl_int err;

//     // 1. Παίρνουμε το πλήθος των διαθέσιμων OpenCL πλατφορμών
//     err = clGetPlatformIDs(0, NULL, &num_platforms);
//     if (err != CL_SUCCESS || num_platforms == 0) {
//         std::cout << "No OpenCL platforms found! Make sure drivers are installed." << std::endl;
//         return 1;
//     }

//     std::vector<cl_platform_id> platforms(num_platforms);
//     clGetPlatformIDs(num_platforms, platforms.data(), NULL);

//     std::cout << "===============================================" << std::endl;
//     std::cout << "    SYSTEM HARDWARE DETECTION (OpenCL API)     " << std::endl;
//     std::cout << "===============================================" << std::endl;
//     std::cout << "Total OpenCL Platforms Found: " << num_platforms << "\n" << std::endl;

//     // 2. Σκανάρουμε κάθε πλατφόρμα για να βρούμε CPU και GPU συσκευές
//     for (cl_uint i = 0; i < num_platforms; ++i) {
//         char platform_name[128];
//         char platform_vendor[128];
//         clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
//         clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        
//         std::cout << "Platform [" << i << "]: " << platform_name << " (" << platform_vendor << ")" << std::endl;

//         cl_uint num_devices;
//         // Ψάχνουμε για οποιοδήποτε τύπο συσκευής (CPU, GPU, κτλ) στη συγκεκριμένη πλατφόρμα
//         err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
//         if (err != CL_SUCCESS || num_devices == 0) {
//             std::cout << "  -> No hardware devices detected on this platform." << std::endl;
//             std::cout << "-----------------------------------------------" << std::endl;
//             continue;
//         }

//         std::vector<cl_device_id> devices(num_devices);
//         clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), NULL);

//         // Εκτύπωση των στοιχείων της κάθε συσκευής
//         for (cl_uint j = 0; j < num_devices; ++j) {
//             char device_name[128];
//             cl_device_type device_type;

//             clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
//             clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);

//             std::cout << "  -> Device [" << j << "]: " << device_name;
            
//             // Έλεγχος του τύπου της συσκευής (Hardware Classification)
//             if (device_type & CL_DEVICE_TYPE_CPU) {
//                 std::cout << " [⚙️ TYPE: CPU]" << std::endl;
//             } else if (device_type & CL_DEVICE_TYPE_GPU) {
//                 std::cout << " [🎮 TYPE: GPU]" << std::endl;
//             } else {
//                 std::cout << " [❓ TYPE: OTHER/ACCELERATOR]" << std::endl;
//             }
//         }
//         std::cout << "-----------------------------------------------" << std::endl;
//     }

//     return 0;
// }

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(err, msg) do { \
    if ((err) != CL_SUCCESS) { \
        printf("Error %d: %s\n", (err), (msg)); \
        exit(1); \
    } \
} while(0)

int main()
{
    cl_int err;

    // -------------------------
    // Get platforms
    // -------------------------
    cl_uint numPlatforms;
    CHECK(clGetPlatformIDs(0, NULL, &numPlatforms),
          "get platform count");

    cl_platform_id *platforms =
        (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

    CHECK(clGetPlatformIDs(numPlatforms, platforms, NULL),
          "get platforms");

    printf("OpenCL Platforms found: %u\n", numPlatforms);

    // -------------------------
    // Loop platforms
    // -------------------------
    for (cl_uint p = 0; p < numPlatforms; p++)
    {
        printf("\n==============================\n");
        printf("Platform %u\n", p);

        // -------------------------
        // Get devices
        // -------------------------
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[p],
                             CL_DEVICE_TYPE_ALL,
                             0, NULL, &numDevices);

        if (err != CL_SUCCESS) {
            printf("No devices on this platform\n");
            continue;
        }

        cl_device_id *devices =
            (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

        CHECK(clGetDeviceIDs(platforms[p],
                             CL_DEVICE_TYPE_ALL,
                             numDevices,
                             devices,
                             NULL),
              "get devices");

        printf("Devices found: %u\n", numDevices);

        // -------------------------
        // Loop devices
        // -------------------------
        for (cl_uint d = 0; d < numDevices; d++)
        {
            printf("\n------------------------------\n");
            printf("Device ID: %u\n", d);

            // Device name
            char name[256];
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_NAME,
                            sizeof(name),
                            name,
                            NULL);
            printf("Name: %s\n", name);

            // Compute units
            cl_uint computeUnits;
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(cl_uint),
                            &computeUnits,
                            NULL);
            printf("Compute Units: %u\n", computeUnits);

            // Global memory
            cl_ulong globalMem;
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(cl_ulong),
                            &globalMem,
                            NULL);

            printf("Global Memory: %.2f GB\n",
                   globalMem / (1024.0 * 1024 * 1024));

            // Local memory
            cl_ulong localMem;
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_LOCAL_MEM_SIZE,
                            sizeof(cl_ulong),
                            &localMem,
                            NULL);

            printf("Local Memory (shared): %zu KB\n",
                   localMem / 1024);

            // Max work-group size
            size_t maxWG;
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(size_t),
                            &maxWG,
                            NULL);
            printf("Max Work-Group Size: %zu\n", maxWG);

            // Max dimensions
            size_t dims[3];
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            sizeof(dims),
                            dims,
                            NULL);

            printf("Max Work Item Sizes: (%zu, %zu, %zu)\n",
                   dims[0], dims[1], dims[2]);

            cl_uint maxDims;
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(cl_uint),
                            &maxDims,
                            NULL);

            printf("Max Dimensions: %u\n", maxDims);

            // Clock frequency
            cl_uint freq;
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            sizeof(cl_uint),
                            &freq,
                            NULL);

            printf("Clock Frequency: %u MHz\n", freq);

            // Vendor
            char vendor[256];
            clGetDeviceInfo(devices[d],
                            CL_DEVICE_VENDOR,
                            sizeof(vendor),
                            vendor,
                            NULL);
            printf("Vendor: %s\n", vendor);
        }

        free(devices);
    }

    free(platforms);

    return 0;
}