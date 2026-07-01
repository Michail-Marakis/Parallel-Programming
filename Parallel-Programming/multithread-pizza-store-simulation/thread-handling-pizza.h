#ifndef HEADER2_H
#define HEADER2_H

#include <pthread.h>
#include <stdbool.h>

//variables
int N_tel = 2;
int N_cook = 2;
int N_oven = 10;
int N_packer = 2;
int N_deliver = 10;

int T_orderlow = 1;
int T_orderhigh = 5;

int N_orderlow = 1;
int N_orderhigh = 5;

int Cm = 12;
int Cp = 14;
int Cs = 15;

int Tprep = 1;
int Tbake = 10;
int Tpack = 1;

int T_paymentlow = 1;
int T_paymenthigh = 3;

int Tdellow = 10;
int Tdelhigh = 15;

float Pm = 45.0;
float Pp = 35.0;
float Ps = 20.0;

float P_fail = 5.0;

//stats
int total_pizzas_sold = 0;
int total_margarita = 0;
int total_special = 0;
int total_peperoni = 0;
int total_revenue = 0;

float synolikos_xronos_ejipiretisis = 0;
float megistos_xronos_ejipiretisis = 0;
float synolikos_xronos_krywmatos = 0;
float megistos_xronos_krywmatos = 0;


int Seed;
int epitixies = 0;
int apotyxies = 0;

//mutexes
pthread_mutex_t mutex_stats = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_prints = PTHREAD_MUTEX_INITIALIZER;

pthread_mutex_t mutex_tel = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_tel = PTHREAD_COND_INITIALIZER;

pthread_mutex_t mutex_cook = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_cook = PTHREAD_COND_INITIALIZER;

pthread_mutex_t mutex_oven = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_oven = PTHREAD_COND_INITIALIZER;

pthread_mutex_t mutex_packer = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_packer = PTHREAD_COND_INITIALIZER;

pthread_mutex_t mutex_deliver = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_deliver = PTHREAD_COND_INITIALIZER;

#endif
