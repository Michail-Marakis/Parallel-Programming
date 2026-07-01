#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>

//#define N 1000
#define N 100000000
#define NUM_THREADS 8

//integration limits
double a = -50.0, b = 50.0;

//step size
double h;

//global sum which is shared between threads
double sum = 0.0;

//initizaling the mutex(lock)
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

double f(double x) {
    // return x * x;
    // return exp(-(x*x));
    return cos(x * x);
}

//thread function
void* compute(void* arg) {
    int id = *(int*)arg;

    //split work equally
    int piece = N / NUM_THREADS;

    int start = id * piece;
    int end;

    //last thread takes remaining part 
    if (id == NUM_THREADS - 1) {
        end = N;
    } else {
        end = (id + 1) * piece;
    }

    double local_sum = 0.0;

    //compute local part first(no need for locking here)
    for (int i = start; i < end; i++) {
        double x = a + i * h;
        local_sum += 2 * f(x);
    }

    //update global sum safely locking the mutex
    pthread_mutex_lock(&mutex);
    sum += local_sum;

    //prograssive printing of the integral
    std::cout << "Current integral value: "
              << sum * h / 2.0 << std::endl;
    //unlock the mutex so the next thread can update the shared variable
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main() {

    h = (b - a) / N;

    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    pthread_mutex_init(&mutex, NULL);

    auto start_time = std::chrono::high_resolution_clock::now();

    //create the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, compute, &ids[i]);
    }

    //wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    //add boundary values
    sum += f(a) + f(b);

    
    double final_res = sum * (h / 2);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Integral = " << final_res << std::endl;
    std::cout << "Time = " << elapsed.count() << " sec\n";

    pthread_mutex_destroy(&mutex);

    return 0;
}