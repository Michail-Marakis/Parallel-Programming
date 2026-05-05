#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define N 100000000

double f(double x) {
    return x * x;
}

// double f(double x) {
//     int k = (int)(1000 * fabs(sin(x)));  
//     double function_sum = 0;
//     for(int i = 0; i < k; i++) {
//         function_sum += exp(-x * x);
//     }
//     return function_sum;
// }


int main() {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;
    double sum = 0.0;

    omp_set_num_threads(1);

    auto start = std::chrono::high_resolution_clock::now();
                                                //eidos,chunk size
    #pragma omp parallel for reduction(+:sum) schedule(static, 1000)
    //#pragma omp parallel for reduction(+:sum) schedule(dynamic, 100)
    //#pragma omp parallel for reduction(+:sum) schedule(guided, 1000)
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