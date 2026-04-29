#include <iostream>
#include <cmath>
#include <chrono>   // for high-resolution timing
#include <omp.h>
#define N 100000000  

double f(double x) {
    return x * x;
}

int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;
    double sum = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel{
        double local_sum = 0.0;

        #pragma omp for
        for (int i = 0; i < N; i++) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            local_sum += (f(x1) + f(x2)) * h / 2.0;
        }

        #pragma omp critical
        sum+= local_sum;
    }


    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Integral from " << a << " to " << b << " = " << sum << std::endl;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}