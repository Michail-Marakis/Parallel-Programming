#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
//iterations
//#define N 100000000
#define N 100000
//functions
double f(double x) {
    
    int k = static_cast<int>(std::abs(std::sin(x * 5.0))) + 50;
    double res = 0.0;
    for (int i = 0; i < k; i++) {
        res += x;
    }
    return res;
}

// double f(double x) {
//     return x * x;
// }


//compute the integral
double integrate_serial(int start, int end, double a, double h) {
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        sum += (f(x1) + f(x2)) * h / 2.0;
    }
    return sum;
}

//recursive 
double task_recursive(int start, int end, double a, double h) {
     
    //check the limit
    if (end - start <= 1000000) {
        return integrate_serial(start, end, a, h);
    }

    int mid = (start + end) / 2;
    double left = 0.0;
    double right = 0.0;
    //new task for left
    #pragma omp task shared(left)
    {
        left = task_recursive(start, mid, a, h);
    }
    //new task for right
    #pragma omp task shared(right)
    {
        right = task_recursive(mid, end, a, h);
    }

    //synchornize
    #pragma omp taskwait
    return left + right;
}

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;
    double final_sum = 0.0;

    int num_threads = 8;

    auto start_time = std::chrono::high_resolution_clock::now();
    //parallel region
    #pragma omp parallel num_threads(num_threads)
    {
        //one thread starts the recursion
        #pragma omp single
        {
            final_sum = task_recursive(0, N, a, h);
        }
        //barrier all threads wait for all the tasks to finish
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Result: " << final_sum << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}