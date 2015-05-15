#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "cuda_interval_lib.h"
#include "helper_timer.h"

#include<omp.h>
#include <boost/numeric/interval.hpp>
using boost::numeric::interval;
using namespace boost::numeric;

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include "helper_cuda.h"

using namespace std;


typedef interval_gpu<double> I_t;

// Kernel
//template <class T>
__global__ void ADD(I_t *a, I_t *b, I_t *c, int N)
{
	int i = blockIdx.x*blockDim.x +threadIdx.x;
	if (i < N)
		c[i] = a[i] + b[i];
}

// Auxiliar
void printI(I_t x);
void gpu_sum(I_t *a, I_t *b, I_t *c, int N);
void read_intervals(I_t *a, I_t *b, int N);

template<class T>
void read_intervals(interval<T> *x,interval<T> *y, int N);

template<class T>
void printI(interval<T> x){
	cout << "[" << lower(x) << ", " << upper(x) << "]" << endl;
}



int main(int argc,char *argv[])
{
	int method = 0;
	int iter = 100;
	bool pflag = false;

	switch(argc){

	case 1:
		break;

	case 4:
		if(!strcmp(argv[3], "-p"))
			pflag = true;

	case 3:
		iter = atoi(argv[2]);

	case 2:
		method = atoi(argv[1]);
		break;

	default:
		method = atoi(argv[1]);
		iter = atoi(argv[2]);
		if(!strcmp(argv[3], "-p"))
			pflag = true;
	}

	int N;
	I_t *a = NULL, *b = NULL, *c = NULL;
	interval<double> *x, *y, *z;

	cin >> N;
	N = N/2;
	if (method >= 2){
		a = new I_t[N];
		b = new I_t[N];
		c = new I_t[N];
		read_intervals(a, b, N);
	}
	else{ 
		x = new interval<double>[N];
		y = new interval<double>[N];
		z = new interval<double>[N];
		read_intervals(x, y, N);
	}
	
	// Run "iter" times and clock the time	
	StopWatchInterface *timer;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	for (int i = 0; i < iter; ++i){
		if (method >= 2){
			gpu_sum(a, b, c, N);
			cudaDeviceSynchronize();
		}
		else if (method == 1){
			omp_set_num_threads(8);
			#pragma omp parallel for schedule(dynamic, 500)
			for (int j = 0; j < N; ++j)
				z[j] = x[j] + y[j];
		}
		else{
			for (int j = 0; j < N; ++j)
				z[j] = x[j] + y[j];
		}
	}
	sdkStopTimer(&timer);

	// print if user asked for it
	if(pflag){
		for (int i = 0; i < N; ++i){
			if (method >= 2)
				printI(c[i]);
			else
				printI(z[i]);
		}
	}
	
	// print execution time
	printf ("Time for the execution of %d interval sums %d times: %f s\n", N, iter, sdkGetTimerValue(&timer)/1000);

	if (method >= 2){
		delete[] a;
		delete[] b;
		delete[] c;
	}
	else {
		delete[] x;
		delete[] y;
		delete[] z;
	}

	return 0;
}

void gpu_sum(I_t* a, I_t* b, I_t* c, int N){

	I_t *d_a, *d_b, *d_c;
	// test malloc time impact (no impact)
	//interval_gpu<double> *z = (interval_gpu<double> *) malloc(N*sizeof(I_t));

	cudaMalloc((void **) &d_a, N*sizeof(I_t));
	cudaMalloc((void **) &d_b, N*sizeof(I_t));
	cudaMalloc((void **) &d_c, N*sizeof(I_t));

	cudaMemcpy(d_a, a, N*sizeof(I_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N*sizeof(I_t), cudaMemcpyHostToDevice);

	// how to set the number of blocks and threads?
	ADD<<<1024, (N+1024)/1024>>>(d_a, d_b, d_c, N);
	//cudaThreadSynchronize();

	cudaMemcpy(c, d_c, N*sizeof(I_t), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void printI(I_t x){
	cout << "[" << x.lower() << ", " << x.upper() << "]" << endl;
}


void read_intervals(I_t* a, I_t* b, int N){

	double aux, aux2;

	for (int i = 0; i < N; ++i){
		cin >> aux; cin >> aux2;
		a[i] = I_t(aux, aux2);
	}
	for (int i = 0; i < N; ++i){
		cin >> aux; cin >> aux2;
		b[i] = I_t(aux, aux2);
	}

}

template<class T>
void read_intervals(interval<T> *x,interval<T> *y, int N){

	double aux, aux2;

	for (int i = 0; i < N; ++i){
		cin >> aux, cin >> aux2;
		x[i] = interval<double>(aux, aux2);
	}
	for (int i = 0; i < N; ++i){
		cin >> aux, cin >> aux2;
		y[i] = interval<double>(aux, aux2);
	}

}

