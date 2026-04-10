#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>

#define N 100000000

#define NUM_THREADS 8

//limits of integration
double a = -50.0;
double b = 50.0;

//step size
double h;

//each thread stores its result here
double table[NUM_THREADS];

double f(double x) {
    // return x * x;
    // return exp(-(x*x));
    return cos(x * x);
}

//thread work function
void* compute(void* arg) {
    int id = *((int*)arg);

    //split work into equal pieces among the threads
    int pieces = N / NUM_THREADS;

    int start_index = id * pieces;

    int end_index;

    //last thread takes the rest if needed
    if (id == NUM_THREADS - 1) {
        end_index = N;
    } else {
        end_index = (id + 1) * pieces;
    }

    double local_sum = 0.0;

    //compute partial sum for this specific thread
    for (int i = start_index; i < end_index; i++) {
        double x = a + i * h;
        local_sum += 2.0 * f(x);
    }

    table[id] = local_sum;

    return NULL;
}

int main() {

    h = (b - a) / N;

    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    auto start_time = std::chrono::high_resolution_clock::now();

    //create the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, compute, &ids[i]);
    }

    //wait all of the threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    double total_sum = 0.0;

    //combine results from each thread
    for (int i = 0; i < NUM_THREADS; i++) {
        total_sum += table[i];

        //prorgessive printing of the integral. 
        std::cout << "Current integral value: "
                  << total_sum * h / 2.0 << std::endl;
    }

    //add the endpoints to complete the trapezoidal rule
    total_sum += f(a) + f(b);

    double result = total_sum * (h / 2.0);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Integral = " << result << std::endl;
    std::cout << "Time = " << elapsed.count() << " seconds" << std::endl;

    return 0;
}