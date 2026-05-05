#include <iostream>
#include <cmath>
#include <chrono>   // for high-resolution timing
#include <omp.h>
#include <string>
#define N 100000000  

double f(double x) {
    return x * x;
}

int main(int argc, char* argv[]) {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;
    double sum = 0.0;
    int num_threads = 1;
    if (argc >= 2) {
        num_threads = std::stoi(argv[1]);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
   
    #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        sum += (f(x1) + f(x2)) * h / 2.0;
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Integral from " << a << " to " << b << " = " << sum << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}