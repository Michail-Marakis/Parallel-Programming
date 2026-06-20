#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "p3230267-p3230111-pizza.h"

void *customers_thread (void *threadId){
	int id = *(int *)threadId;										//we pass each customer's id to the method
	free(threadId);													//free memory from the variable

	struct timespec start_time_customer_service1, end_time_customer_service1;
	clock_gettime(CLOCK_REALTIME, &start_time_customer_service1);	//clock for service time

	int cnt_tel = 0;
	struct timespec start_time, end_time; 
	clock_gettime(CLOCK_REALTIME, &start_time); 					//helper clock for customer's waiting duration for a telephone operator

	pthread_mutex_lock(&mutex_tel); 								//mutex lock for telephone operator
	while (N_tel == 0) { 											//check case of no available telephone operators
		cnt_tel++;
		if (cnt_tel == 1) { 										//condition to print only once
			pthread_mutex_lock(&mutex_prints); 						//mutex lock to print that there is no available telephone operator
			printf("Customer with order number %d is waiting for a telephone operator to become available.\n", id);	
			pthread_mutex_unlock(&mutex_prints);					//mutex unlock for printing
		}
		pthread_cond_wait(&cond_tel, &mutex_tel);  					//signal for next threads to wait
	}
	N_tel--;														//reserve 1 telephone operator
	pthread_mutex_unlock(&mutex_tel);								//mutex unlock for telephone operator

	clock_gettime(CLOCK_REALTIME, &end_time); 						//closing helper clock
	float waitingTime_order = end_time.tv_sec - start_time.tv_sec;

	pthread_mutex_lock(&mutex_prints);							//mutex lock to print customer's waiting time to order
	printf("Customer with order number %d waited %.2f minutes before placing the order.\n", id, waitingTime_order);
	pthread_mutex_unlock(&mutex_prints);						//mutex unlock for printing

	unsigned int seed = Seed + time(NULL);						//initialize Seed so it is different in each thread

	int num_pizzas = rand_r(&seed) % (N_orderhigh - N_orderlow + 1) + N_orderlow;		//calculate number of pizzas
	int margarita = 0, peperoni = 0, special = 0;

	for (int i = 0; i < num_pizzas; i++) {  					//calculate each pizza type according to the statement probabilities
		float pizza_type_chance = rand_r(&seed) % 101;
		if (pizza_type_chance <= Pm)
			margarita++;					//depending on which numerical interval the random number falls into, the pizza type is decided
		else if (pizza_type_chance <= Pm + Pp)
			peperoni++;
		else
			special++;
	}

	int paying_delay = rand_r(&seed) % (T_paymenthigh - T_paymentlow + 1) + T_paymentlow;			//calculate delay
	sleep(paying_delay);										//sleep the thread for the delay interval

	float NoMoney_chance = rand_r(&seed) % 101;					//probability of order cancellation due to insufficient funds


	if (NoMoney_chance <= P_fail) { 		//calculate statistics in case the order fails due to insufficient funds
		pthread_mutex_lock(&mutex_stats);
		apotyxies++;
		pthread_mutex_unlock(&mutex_stats);

		pthread_mutex_lock(&mutex_prints);
		printf("Order with number %d failed.\n", id);	//lock and unlock the appropriate mutexes for stats and printing
		pthread_mutex_unlock(&mutex_prints);

		pthread_mutex_lock(&mutex_tel);
		N_tel++;												//release the telephone operator
		pthread_cond_signal(&cond_tel);
		pthread_mutex_unlock(&mutex_tel);

		pthread_exit(NULL);										//exit the customer from the program as the process is finished
	} else {
		pthread_mutex_lock(&mutex_stats);
		epitixies++;
		pthread_mutex_unlock(&mutex_stats);

		pthread_mutex_lock(&mutex_prints);
		printf("Order with number %d was registered.\n", id);
		pthread_mutex_unlock(&mutex_prints);
	}

	clock_gettime(CLOCK_REALTIME, &end_time_customer_service1); 				//end clock for service time until registration
	float waitingTime_customer_service = end_time_customer_service1.tv_sec - start_time_customer_service1.tv_sec;

	struct timespec start_time_customer_service2, end_time_customer_service2;		//clock for service
	clock_gettime(CLOCK_REALTIME, &start_time_customer_service2);

	struct timespec start_packing, end_packing; 							//clock until packing
	clock_gettime(CLOCK_REALTIME, &start_packing);
	
	struct timespec start_delivery, end_delivery; 						//clock until delivery
	clock_gettime(CLOCK_REALTIME, &start_delivery);

	pthread_mutex_lock(&mutex_stats); 									//mutex lock for statistics calculation
	total_pizzas_sold += num_pizzas;
	total_margarita += margarita;
	total_peperoni += peperoni;
	total_special += special;
	total_revenue += margarita * Cm + peperoni * Cp + special * Cs;
	pthread_mutex_unlock(&mutex_stats);

	pthread_mutex_lock(&mutex_tel);
	N_tel++; 														//release telephone operator
	
	pthread_cond_signal(&cond_tel); 					//notify the next waiting thread that a telephone operator was released
	pthread_mutex_unlock(&mutex_tel);


	int cnt_cook = 0;
	pthread_mutex_lock(&mutex_cook); 								//mutex lock for cooks
	while (N_cook == 0) {									 //check case of no available cooks
		cnt_cook++;
		if(cnt_cook == 1){
			pthread_mutex_lock(&mutex_prints);	//mutex lock and print once the message if there is no available cook
			printf("Order with number %d is waiting for a cook to become available.\n",id);
			pthread_mutex_unlock(&mutex_prints);
		}
		pthread_cond_wait(&cond_cook, &mutex_cook); 			//signal for next threads to wait
	}
	N_cook--; 													//reserve the cook
	pthread_mutex_unlock(&mutex_cook); 							//mutex unlock for cooks

	sleep(Tprep);												//create delay for pizza preparation time

	pthread_mutex_lock(&mutex_cook);	
	N_cook++;													//release the cook
	pthread_cond_signal(&cond_cook);					//notify the next waiting thread that a cook was released
	pthread_mutex_unlock(&mutex_cook);
	
	pthread_mutex_lock(&mutex_oven);						//mutex lock for ovens
	int cnt_oven = 0;
	while (N_oven == 0 || N_oven - num_pizzas < 0) { 		//check oven unavailability in case the order does not fit in available ovens
		cnt_oven++;
		if(cnt_oven == 1){
			pthread_mutex_lock(&mutex_prints); 			//mutex lock and print once the message if there are no available ovens and how many are needed at that moment for each order
			printf("Order with number %d is currently waiting for %d ovens to become available.\n", id, num_pizzas - N_oven);
			pthread_mutex_unlock(&mutex_prints);					//mutex unlock
		}
		pthread_cond_wait(&cond_oven, &mutex_oven); 				//signal for next threads to wait
	}
	N_oven -= num_pizzas;											//reserve 1 oven per pizza
	pthread_mutex_unlock(&mutex_oven);								//mutex unlock


	sleep(Tbake);											//create delay for baking time

	struct timespec start_time_cold, end_time_cold; 		//clock to calculate cooling time
	clock_gettime(CLOCK_REALTIME, &start_time_cold);


	int cnt_packer = 0;
	pthread_mutex_lock(&mutex_packer);						//packer mutex
	while (N_packer == 0) {									//check case of no available packers
		cnt_packer++;		
		if(cnt_packer == 1){
			pthread_mutex_lock(&mutex_prints); 		//mutex lock and print once the message if there is no available packer
			printf("Order with number %d is waiting for a packing employee to become available.\n",id);
			pthread_mutex_unlock(&mutex_prints); 			//mutex unlock
		}	 
		pthread_cond_wait(&cond_packer, &mutex_packer);		//signal for next threads to wait
	}
	N_packer--;
	pthread_mutex_unlock(&mutex_packer); 	//mutex unlock for packer

	sleep(Tpack); 							//create delay for packing time

	pthread_mutex_lock(&mutex_packer);
	N_packer++;								//release packer
	pthread_cond_signal(&cond_packer);		//notify the next thread to wake up
	pthread_mutex_unlock(&mutex_packer);

		
	clock_gettime(CLOCK_REALTIME, &end_packing); 							//end packing time clock
	float packing_time = end_packing.tv_sec - start_packing.tv_sec;


	float totalWaitingPacking = packing_time + waitingTime_customer_service; 		//calculate total preparation time
	pthread_mutex_lock(&mutex_prints);
	printf("Order with number %d was prepared in %.2f minutes.\n", id, totalWaitingPacking);
	pthread_mutex_unlock(&mutex_prints);

	pthread_mutex_lock(&mutex_oven); 					//release ovens
	N_oven+= num_pizzas; 
	pthread_cond_broadcast(&cond_oven); //broadcast to wake up all threads so the first order that fits and arrives can enter
	pthread_mutex_unlock(&mutex_oven);

	int cnt_deliver = 0;
	pthread_mutex_lock(&mutex_deliver); 					//deliverer mutex
	while (N_deliver == 0) { 						//check case of no available deliverers
		cnt_deliver++;
		if(cnt_deliver == 1){
			pthread_mutex_lock(&mutex_prints);		//mutex lock and print once the message that there are no available deliverers
			printf("Order with number %d is waiting for a delivery driver to become available.\n",id);
			pthread_mutex_unlock(&mutex_prints); 		//mutex unlock for printing
		}	
		pthread_cond_wait(&cond_deliver, &mutex_deliver); 			//signal for next threads to wait
	}
	N_deliver--; 													//reserve deliverer
	pthread_mutex_unlock(&mutex_deliver); 						//mutex unlock for deliverer


	int delivery_duration = rand_r(&seed) % (Tdelhigh - Tdellow + 1) + Tdellow;		//set delivery time
	sleep(delivery_duration); 									//deliverer time to take the order

	clock_gettime(CLOCK_REALTIME, &end_delivery); 					//end delivery clock
	float actual_delivery_time = end_delivery.tv_sec - start_delivery.tv_sec;

	float TotalWaitingDelivery = actual_delivery_time + waitingTime_customer_service; 		//total delivery time
	pthread_mutex_lock(&mutex_prints);								//mutex lock for printing
	printf("Order with number %d was delivered in %.2f minutes.\n", id, TotalWaitingDelivery);
	pthread_mutex_unlock(&mutex_prints); 							//mutex unlock for printing

	clock_gettime(CLOCK_REALTIME, &end_time_customer_service2); //close service clock
	clock_gettime(CLOCK_REALTIME, &end_time_cold); //close cooling clock

	float waitingTime_customer_service_2nd_phase = end_time_customer_service2.tv_sec - start_time_customer_service2.tv_sec;

	float total_customer_service = waitingTime_customer_service + waitingTime_customer_service_2nd_phase;

	if (total_customer_service > megistos_xronos_ejipiretisis) { //calculate maximum service time
		megistos_xronos_ejipiretisis = total_customer_service;
	}
	synolikos_xronos_ejipiretisis += total_customer_service;//sum total service time to calculate average

	float waitingTime_cold = end_time_cold.tv_sec - start_time_cold.tv_sec;

	if (waitingTime_cold > megistos_xronos_krywmatos) {//calculate maximum cooling time
		megistos_xronos_krywmatos = waitingTime_cold;
	}
	synolikos_xronos_krywmatos += waitingTime_cold;//sum total cooling time to calculate average

	sleep(delivery_duration); 					//deliverer return

	pthread_mutex_lock(&mutex_deliver);
	N_deliver++; 								//release deliverer
	pthread_cond_signal(&cond_deliver);
	pthread_mutex_unlock(&mutex_deliver);

	pthread_exit(NULL);		//exit the thread/customer from the process
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Not enough arguments provided\n");
        exit(0);
    }
 
    int Ncust = atoi(argv[1]); 					//argument 1 (number of customers)
    Seed = atoi(argv[2]);						//argument 2 (seed)
 
    pthread_t* customers = malloc(Ncust * sizeof(pthread_t));
    if (customers == NULL) {
		printf("Not enough memory available\n");		//check if there is enough memory to allocate for threads
        exit(-1);
    }
 
    for (int i = 0; i < Ncust; i++) {
        int* id = malloc(sizeof(int));
        if (id == NULL) {						//check if there is enough memory to allocate for ids
            printf("Not enough memory available\n");
            exit(-1);
        }
 
        *id = i + 1;
 		int rc = pthread_create(&customers[i], NULL, customers_thread, id); //create threads, pass to customers_thread method and check if an error occurs
        if (rc != 0) {
            printf("Error during thread (customer) creation\n");
            free(id);
            exit(0);
        }
 		
 		unsigned int seedp = Seed + i ; 				//we accept the seed
        int kathisterisi = rand_r(&seedp) % (T_orderhigh - T_orderlow +1) + T_orderlow; 
        sleep(kathisterisi);
    }
 
    for (int i = 0; i < Ncust; i++) { 				//wait for all threads to finish before printing final statistics
        pthread_join(customers[i], NULL);
    }

	printf("Total revenue: %d $ \n", total_revenue);

	printf("Number of Margarita pizzas sold: %d \n", total_margarita);
	
	printf("Number of Pepperoni pizzas sold: %d \n", total_peperoni);
	
	printf("Number of Special pizzas sold: %d \n", total_special);
	
	printf("Total number of pizzas sold: %d \n", total_margarita + total_peperoni + total_special);

										//printing final statistics.
	printf("Successful orders: %d while failed orders: %d \n", epitixies, apotyxies);
	
	printf("Maximum service time for successful orders was: %.2f minutes\n",megistos_xronos_ejipiretisis);
	
	if(epitixies!=0){					//check for division by 0 case
		printf("Average service time for successful orders was: %.2f minutes \n", synolikos_xronos_ejipiretisis/epitixies);
		printf("Average cooling time of orders was: %.2f minutes\n",synolikos_xronos_krywmatos/epitixies);
	}else{
		printf("All orders failed. \n");
	}
	printf("Maximum cooling time of orders was: %.2f minutes\n",megistos_xronos_krywmatos);
	
    free(customers);					//free memory
    return 0;
}
