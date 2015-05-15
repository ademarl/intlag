
// BLAS adaptation for interval_gpu by means of box_gpu

// TODO: Rotations
// TODO: Increments
// TODO: One BLAS2 method

#ifndef BOX_BLAS_H
#define BOX_BLAS_H

#include "cuda_box_lib.h"

//////////////////////////////////////////// Auxiliar //////////////////////////

// Loops through kernels which calculate the sum of adjacent pairs, storing
//them on temporary memory, then swaps the references to these memory locations
template <class T> inline __host__
void asum_loop(int N, DeviceData<T>& x) {

	DeviceData< interval_gpu<double> > y((N+1)/2);

	while(N != 1) {
		int n = (N+1)/2;
		ASUM<<<blocks(n), threads(n)>>>(x.data(), y.data(), N);
		cudaDeviceSynchronize();
		swapByReference(x, y);
		N = n;
	}
}

//////////////////////////////////////////// ~Auxiliar /////////////////////////


// Copies x into y
template<class T> inline __host__
bool acopy(box_gpu<T> const &x, box_gpu<T> &y) {
	y = x;
	return true;
}


// Swap
template<class T> inline __host__
bool swap(box_gpu<T> &x, box_gpu<T> &y) {

	if (&x == &y) return false;
	if (x.length() != y.length()) return false;

	box_gpu<T> aux;
	aux = x;
	x = y;
	y = aux;

	return true;
}



// Scal
template<class T> inline __host__
bool scal(T alpha, box_gpu<T> &x) {
	x = alpha*x;
	return true;
}


// AXPY: alpha*x + y
template<class T> inline __host__
bool axpy(T alpha, box_gpu<T> &x, box_gpu<T> const &y) {
	
	int N = x.length();
	DeviceData< interval_gpu<double> > d_x(N, x.data()), d_y(N, y.data());

	AXPY<<<blocks(N), threads(N)>>>(alpha, d_x.data(), d_y.data(), d_x.data(), N);

  box_gpu<T> ret(N);
	d_x.toHost(x.data());
	return true;
}


// Sum of all elements
template<class T> inline __host__
bool asum(box_gpu<T> const &x, interval_gpu<T> *sum) {
	
	int N = x.length();
	DeviceData< interval_gpu<T> > d_x(N, x.data());

	asum_loop(N, d_x);

	d_x.toHost(sum, 1);

	return true;
}


// Dot Product
template<class T> inline __host__
bool dot(box_gpu<T> const &x, box_gpu<T> const &y, interval_gpu<T> *dot) {
	
	int N = x.length();
	DeviceData< interval_gpu<T> > d_x(N, x.data()), d_y(N, y.data());

	// Calculate Hadamard product
	MULT<<<blocks(N), threads(N)>>>(d_x.data(), d_y.data(), d_x.data(), N);
	CHECKED_CALL( cudaDeviceSynchronize() );

	// Now sum all elements
	asum_loop(N, d_x);

	d_x.toHost(dot, 1);
	return true;
}


// Norm2
template<class T> inline __host__
bool norm2(box_gpu<T> const &x, interval_gpu<T> *norm2) {
	
	int N = x.length();
	DeviceData< interval_gpu<T> > d_x(N, x.data());

	// Calculate Hadamard product
	MULT<<<blocks(N), threads(N)>>>(d_x.data(), d_x.data(), d_x.data(), N);
	CHECKED_CALL( cudaDeviceSynchronize() );

	// Now sum all elements
	asum_loop(N, d_x);

	// Then, take the sqrt
	SQRT<<<1,1>>>((d_x.data()), d_x.data());

	d_x.toHost(norm2, 1);
	return true;
}




#endif
