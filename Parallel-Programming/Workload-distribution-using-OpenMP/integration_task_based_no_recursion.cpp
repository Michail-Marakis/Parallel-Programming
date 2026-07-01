#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
//iterations
//#define N 100000000
#define N 100000

//functions
// double f(double x) {
    
//     int k = static_cast<int>(std::abs(std::sin(x * 5.0))) + 50;
//     double res = 0.0;
//     for (int i = 0; i < k; i++) {
//         res += x;
//     }
//     return res;
// }

double f(double x) {
    return x * x;
}

//calculation of the integral 
double integrate_segment(int start, int end, double a, double h) {
    double partial_sum = 0.0;
    for (int i = start; i < end; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        partial_sum = partial_sum + (f(x1) + f(x2)) * h / 2.0;
    }
    return partial_sum;
}

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;
    double total_sum = 0.0;

    int num_threads = 8; //number of threads
    int num_tasks = 128; //number of tasks
    //chunk size
    int chunk_size = N / num_tasks;

    auto start_time = std::chrono::high_resolution_clock::now();
    //parallel region
    #pragma omp parallel num_threads(num_threads)
    {
        //one thread is doing tasks
        #pragma omp single
        {
            for (int i = 0; i < num_tasks; i++) {
                int start_idx = i * chunk_size;
                int end_idx;
                //limit for every task
                if (i == num_tasks - 1) {
                    end_idx = N;
                } else {
                    end_idx = start_idx + chunk_size;
                }
                //define the firstprivate and shared variables
                //firstprivate to copy current values of variables inside the task
                //shared for all threads to have access in the same variable
                #pragma omp task firstprivate(start_idx, end_idx, a, h) shared(total_sum)
                {
                    double result = integrate_segment(start_idx, end_idx, a, h);
                    //atomic for safe update of global sum
                    #pragma omp atomic
                    total_sum = total_sum + result;
                }
            }
        } 

        //barrier that wait for all tasks to be completed
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Result: " << total_sum << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}