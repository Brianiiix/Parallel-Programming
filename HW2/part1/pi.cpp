/*****************************************************************************

	Super-Fast MWC1616 Pseudo-Random Number Generator
	for Intel/AMD Processors (using SSE or SSE4 instruction set)
	Copyright (c) 2012, Ivan Dimkovic
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.

	Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation
	and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
	THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
	PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
	CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
	EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
	OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <random>
#include "FastRand.h"
using namespace std;

unsigned long long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
}

long long int number_in_circle = 0;
pthread_mutex_t lock;

typedef struct
{
    int thread_id;
    int start;
    int end;
    unsigned int seed;
} thread_arg;

void* monte_carlo(void* arg){
    thread_arg *data = (thread_arg *)arg;
    int thread_id = data->thread_id;
    int start = data->start;
    int end = data->end;
    long long int local_in_circle = 0;
    unsigned int seed = data->seed;

/*
    fastrand fr;
	uint32_t prngSeed[16];
	uint16_t *sptr = (uint16_t *)prngSeed;
	//
	// Randomize the seed values
    //cout<<thread_id<<endl;
    //srand(seed);
	for(uint8_t i=0; i<16; i++)
	{
        //cout<<"start"<<endl;
        //srand(seed);
		prngSeed[i] = rand_r(&seed);
        //cout << prngSeed[i] << endl;
	}
	//
	// Initialize the PRNG
	InitFastRand(	sptr[0], sptr[1],
		sptr[2], sptr[3],
		sptr[4], sptr[5],
		sptr[6], sptr[7],
		sptr[8], sptr[9],
		sptr[10], sptr[11],
		sptr[12], sptr[13],
		sptr[14], sptr[15],
		&fr);

    for(int toss = 0; toss < (end-start)/2; toss ++){
        double res[4];
		FastRand_SSE4(&fr);
		res[0] = (double)fr.res[0] / 4294967295.0;
		res[1] = (double)fr.res[1] / 4294967295.0;
        res[2] = (double)fr.res[2] / 4294967295.0;
        res[3] = (double)fr.res[3] / 4294967295.0;
        //printf("%f\n%f\n%f\n%f\n", res[0],res[1],res[2],res[3]);
        if(res[0] * res[0] + res[1] * res[1] <= 1.0)
            local_in_circle ++;
        if(res[2] * res[2] + res[3] * res[3] <= 1.0)
            local_in_circle ++;
    }
    if((end - start) % 2 != 0){
        double res[4];
		FastRand_SSE4(&fr);
        res[0] = (double)fr.res[0] / 4294967295.0;
		res[1] = (double)fr.res[1] / 4294967295.0;
        if(res[0] * res[0] + res[1] * res[1] <= 1.0)
            local_in_circle ++;
    }
*/


    for(int toss = start; toss < end; toss ++){
        double x = (double)rand_r(&seed)/RAND_MAX;
        double y = (double)rand_r(&seed)/RAND_MAX;
        double z = x*x + y*y;
        if(z <= 1.0)
            local_in_circle ++;
    }



    pthread_mutex_lock(&lock);
    number_in_circle += local_in_circle;
    pthread_mutex_unlock(&lock);

    pthread_exit(NULL);
}

int main(int argc, char** argv){
    if(argc != 3){
        cout << "Number of argument must be 2.";
        return 0;
    }
    int thread_num = atoi(argv[1]);
    long long int toss_num = atoll(argv[2]);

    pthread_t thread[thread_num];
    pthread_mutex_init(&lock, NULL);

    int part = toss_num / thread_num;
    thread_arg arg[thread_num];
    srand(time(NULL));
    for (int i = 0; i < thread_num; i++)
    {
        arg[i].thread_id = i;
        arg[i].start = part * i;
        if(i != thread_num - 1) arg[i].end = part * (i + 1);
        else arg[i].end = toss_num;
        arg[i].seed = rdtsc();
        pthread_create(&thread[i], NULL, monte_carlo, (void *)&arg[i]);
    }

    for(int i = 0;  i < thread_num; i ++){
        pthread_join(thread[i], NULL);
    }
    pthread_mutex_destroy(&lock);

    double pi_estimate = 4 * (double)number_in_circle / (double)toss_num;
    cout << pi_estimate << endl;
    return 0;
}
