
//------------------------------------------------------------------------------
// Copyright (c) 2013 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef INTERVAL_TEST
#define INTERVAL_TEST

#include "stdio.h"
#include "gtest/gtest.h"

#include "gpu_interval.h"
#include "gpu_interval/cuda_interval_lib.h"

using namespace intlag;

typedef interval_gpu<float> If_t;
typedef interval_gpu<double> Id_t;



//////////////////////////////////////////////////////////// Kernels ///////////
template <class T>
__global__ void add(T a, T b, T *c){
	*c = a + b;
}

template <class T>
__global__ void scalL(double alpha, T x, T *y){
	*y = alpha*x;
}

template <class T>
__global__ void scalR(double alpha, T x, T *y){
	*y =x*alpha;
}

template <class T>
__global__ void fma(double alpha, T x, T y, T* z){
	*z = fma(alpha, x, y);
}

template <class T>
__global__ void sqrt_test(interval_gpu<T> x, interval_gpu<T>* y){
		// check for signal of upper and lower extremes
		*y = sqrt(x);
}
///////////////////////////////////////////////////////////// ~Kernels /////////


////////////////////////////////////////////////////////// CUDA Functions //////
template <class T>
void sum_intervals(T a, T b, T *c){

	T *d_c;
	CHECKED_CALL( cudaMalloc((void**)&d_c, sizeof(T)) );
	add<<<1,1>>>(a, b, d_c);
	CHECKED_CALL( cudaMemcpy(c, d_c, sizeof(T), cudaMemcpyDeviceToHost) );
	CHECKED_CALL( cudaFree(d_c) );
}

template <class T>
void scalL_intervals(double alpha, T x, T *y){

	T *d_y;
	CHECKED_CALL( cudaMalloc((void**)&d_y, sizeof(T)) );
	scalL<<<1,1>>>(alpha, x, d_y);
	CHECKED_CALL( cudaMemcpy(y, d_y, sizeof(T), cudaMemcpyDeviceToHost) );
	CHECKED_CALL( cudaFree(d_y) );
}


template <class T>
void scalR_intervals(double alpha, T x, T *y){

	T *d_y;
	CHECKED_CALL( cudaMalloc((void**)&d_y, sizeof(T)) );
	scalR<<<1,1>>>(alpha, x, d_y);
	CHECKED_CALL( cudaMemcpy(y, d_y, sizeof(T), cudaMemcpyDeviceToHost) );
	CHECKED_CALL( cudaFree(d_y) );
}

template <class T>
void fma_intervals(double alpha, T a, T b, T *c){

	T *d_c;
	CHECKED_CALL( cudaMalloc((void**)&d_c, sizeof(T)) );
	fma<<<1,1>>>(alpha, a, b, d_c);
	CHECKED_CALL( cudaMemcpy(c, d_c, sizeof(T), cudaMemcpyDeviceToHost) );
	CHECKED_CALL( cudaFree(d_c) );
}

template <class T>
void sqrt_intervals(T a, T *b){

	T *d_b;
	CHECKED_CALL( cudaMalloc((void**)&d_b, sizeof(T)) );
	sqrt_test<<<1,1>>>(a, d_b);
	CHECKED_CALL( cudaMemcpy(b, d_b, sizeof(T), cudaMemcpyDeviceToHost) );
	CHECKED_CALL( cudaFree(d_b) );
}

////////////////////////////////////////////////////////// ~CUDA Functions /////

class IntervalTest : public ::testing::Test{

	protected:

	virtual void SetUp(){
		x = Id_t(-2.1, 3.5);		
		y = Id_t(3.0, 5.5);
		z = Id_t();
	}

	Id_t x, y, z;
};

TEST_F(IntervalTest, Create) {

	EXPECT_DOUBLE_EQ(x.lower(), -2.1);
	EXPECT_DOUBLE_EQ(x.upper(), 3.5);

};


TEST_F(IntervalTest, Addition) {

	sum_intervals(x, y, &z);
	EXPECT_DOUBLE_EQ(0.9, z.lower());
	EXPECT_DOUBLE_EQ(9.0, z.upper());

};

TEST_F(IntervalTest, ScalMultL) {

	scalL_intervals(0.5, x, &z);
	EXPECT_DOUBLE_EQ(-1.05, z.lower());
	EXPECT_DOUBLE_EQ(1.75, z.upper());

};

TEST_F(IntervalTest, ScalMultR) {

	scalR_intervals(-0.5, x, &z);
	EXPECT_DOUBLE_EQ(-1.75, z.lower());
	EXPECT_DOUBLE_EQ(1.05, z.upper());

};

TEST_F(IntervalTest, FMA) {

	fma_intervals(-0.5, x, y, &z);
	EXPECT_DOUBLE_EQ(1.25, z.lower());
	EXPECT_DOUBLE_EQ(6.55, z.upper());

};

TEST_F(IntervalTest, SQRT) {

	sqrt_intervals(y, &z);
	EXPECT_NEAR(1.732, z.lower(), 0.1);
	EXPECT_NEAR(2.345, z.upper(), 0.1);

};

#endif
