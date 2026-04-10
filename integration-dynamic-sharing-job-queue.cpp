#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>

#define N 100000000
//#define N 1000

// number of tasks (pieces of work)
#define K 100

#define NUM_THREADS 4

//integration limits
double a = 0.0, b = 10.0;

//step size
double h;

//global result. the shared variable
double sum = 0.0;

//task counter, needed for dynamic scheduling
int taskid = 0;

//mutex for protecting shared sum
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

//mutex for task assignment
pthread_mutex_t tmutex = PTHREAD_MUTEX_INITIALIZER;


double f(double x) {
    return x * x;
    // return exp(-(x*x));
    // return cos(x*x);
}

//thread function
void* compute(void* arg) {
    int id = *(int*)arg;

    //number of tasks total
    int NTASK = N / K;

    int t;

    //each thread keeps taking new tasks dynamically
    while (true) {

        //get next available task safely locking the mutex
        pthread_mutex_lock(&tmutex);
        t = taskid++;

        //unlocking the mutex for the next threads 
        pthread_mutex_unlock(&tmutex);

        //no more tasks
        if (t >= NTASK) break;

        double local_sum = 0.0;

        //process piece of size K
        for (int i = t * K; i < (t + 1) * K; i++) {
            double x = a + i * h;
            local_sum += 2 * f(x);
        }

        //update global sum safely
        //locking the mutex
        pthread_mutex_lock(&mutex);
        sum += local_sum;

        //unlocking the mutex so other threads can update on the shared variable
        pthread_mutex_unlock(&mutex);
    }

    return NULL;
}

int main() {

    h = (b - a) / N;

    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&tmutex, NULL);

    auto start_time = std::chrono::high_resolution_clock::now();

    //create all the threads
    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, compute, &ids[i]);
    }

    //wait for all the threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    //add boundary points for the trapezoidal rule
    sum += f(a) + f(b);

    double final_res = sum * (h / 2);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Integral = " << final_res << std::endl;
    std::cout << "Time = " << elapsed.count() << " sec\n";

    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&tmutex);

    return 0;
}