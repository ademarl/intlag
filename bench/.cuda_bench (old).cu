
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BENCH_H
#define CUDA_BENCH_H

#include <fstream>

#include "aux/io.h"

#include "aux/case.h"
#include "../include/blas/cuda_blas.h"
#include "gtest/gtest.h"

namespace intlag
{
namespace bench
{

//----------------------- Fixture --------------------------------------------//

class CudaIntervalBench : public BenchTest {
  public:

    CudaIntervalBench() {
      double aux, aux2;

	    cin >> dalpha; alpha = (float)dalpha;
      cin >> dbeta; beta = (float)dbeta;

      cin >> N; N = N/2;
      x = (CudaInterval<float>*) malloc(N*sizeof(CudaInterval<float>));
      y = (CudaInterval<float>*) malloc(N*sizeof(CudaInterval<float>));
      dx = (CudaInterval<double>*) malloc(N*sizeof(CudaInterval<double>));
      dy = (CudaInterval<double>*) malloc(N*sizeof(CudaInterval<double>));

	    for (int i = 0; i < N; ++i){
		    cin >> aux; cin >> aux2;
        x[i] = CudaInterval<float>((float)aux, (float)aux2);
		    dx[i] = CudaInterval<double>(aux, aux2);
	    }

	    for (int i = 0; i < N; ++i){
		    cin >> aux; cin >> aux2;
        y[i] = CudaInterval<float>((float)aux, (float)aux2);
		    dy[i] = CudaInterval<double>(aux, aux2);
      }
    }
    CudaIntervalBench(int a) {}
    virtual ~CudaIntervalBench() {}

    void SetUp() {}
    void TearDown() {}


    // Object Members
    static int N;
    static float alpha, beta;
    static double dalpha, dbeta;
    static CudaInterval<float> *x, *y;
    static CudaInterval<double> *dx, *dy;
};


int CudaIntervalBench::N;
float CudaIntervalBench::alpha, CudaIntervalBench::beta;
double CudaIntervalBench::dalpha, CudaIntervalBench::dbeta;
CudaInterval<float>* CudaIntervalBench::x;
CudaInterval<float>* CudaIntervalBench::y;
CudaInterval<double>* CudaIntervalBench::dx;
CudaInterval<double>* CudaIntervalBench::dy;


//----------------------- Scal Bench -----------------------------------------//

class CudaIntervalBenchScal : public CudaIntervalBench  {
   public:

    CudaIntervalBenchScal(int a = 0) {
    }
    
    void begin() {
      EXPECT_FLOAT_EQ(CudaIntervalBench::alpha, -1.295118);
    }
    void run()   {
      EXPECT_EQ(-2, -2);
    }
    void end()   {
      EXPECT_EQ(2, 2);
    }
};


BENCH_F(CudaIntervalBench, Scal, CudaIntervalBenchScal) // Magic!


//----------------------- Scal Bench -----------------------------------------//



} // namespace bench
} // namespace intlag

#endif



