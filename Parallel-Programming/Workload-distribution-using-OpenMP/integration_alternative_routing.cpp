#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
//iterations
//#define N 100000000
#define N 100000

//functions
double f(double x) {
    return x * x;
}

// double f(double x) {
//     int k = (int)(x * 10);
//     double total = 0.0;
//     for (int i = 0; i < k; i++) {
//         total += x;
//     }
//     return total;
// }

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;
    double sum = 0.0;
    //number of threads
    
    int num_threads = 8;
    auto start = std::chrono::high_resolution_clock::now();
                                                //method,chunk size
    //#pragma omp parallel for reduction(+:sum) schedule(static, 1000)
    //#pragma omp parallel for reduction(+:sum) schedule(static, 1)
    #pragma omp parallel for reduction(+:sum) schedule(guided, 1000) num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        sum += (f(x1) + f(x2)) * h / 2.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Result: " << sum << std::endl;
    std::cout << "Time: " << elapsed.count() << std::endl;

    return 0;
}