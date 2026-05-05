#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <string>

#define N 100000000
using namespace std;

double f(double x) {
    //return x * x;
    return exp(-(x*x));
}

int main(int argc, char* argv[]) {
    double a = 0.0, b = 10.0;
    double h = (b - a) / N;
    double sum = 0.0;

    omp_set_dynamic(0);        
    int num_threads = 1;
    if (argc >= 2) {
        num_threads = std::stoi(argv[1]);
    }
        

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel firstprivate(h, a) num_threads(num_threads)
    {
        double local_sum = 0.0;

        #pragma omp for
        for (int i = 0; i < N; i++) {
            double x1 = a + i * h;
            double x2 = a + (i + 1) * h;
            local_sum += (f(x1) + f(x2)) * h / 2.0;
        }

        #pragma omp critical
        sum += local_sum;
    }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << "Integral from " << a << " to " << b << " = " << sum << endl;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    return 0;
}