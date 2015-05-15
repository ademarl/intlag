
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef SERIAL_BLAS3_TEST_CU
#define SERIAL_BLAS3_TEST_CU

#include <math.h>
#include "blas/serial_blas.h"
#include "aux/test_interval.h"
#include "aux/reference.h"
#include "gtest/gtest.h"

#include "aux/ulp.h"


using namespace intlag;


class SerialBlasTest : public ::testing::Test {
  protected:

    SerialBlasTest() {
      r = Reference::getInstance();
    }
    virtual ~SerialBlasTest(){}

    virtual void SetUp()
    {
      for(int i = 0; i < 9; ++i) {
          a[i] = Interval<float>(i);
          b[i] = Interval<float>(-i, i);
          c[i] = Interval<float>(i-5, i+i*i);
      }
      for(int i = 0; i < 10000; ++i) {
          x[i] = Interval<float>(i);
          y[i] = Interval<float>(-i, i);
          z[i] = Interval<float>(i-5000, i+i*i);
      }

      for(int i = 0; i < 9; ++i) {
          da[i] = Interval<double>(i);
          db[i] = Interval<double>(-i, i);
          dc[i] = Interval<double>(i-5, i+i*i);
      }
      for(int i = 0; i < 10000; ++i) {
          dx[i] = Interval<double>(i);
          dy[i] = Interval<double>(-i, i);
          dz[i] = Interval<double>(i-5000, i+i*i);
      }
    }
    virtual void TearDown() {}

    Reference* r;

    Interval<float> a[9], b[9], c[9];
    Interval<float> x[10000], y[10000], z[10000];
    Interval<double> da[9], db[9], dc[9], dx[10000], dy[10000], dz[10000];
};




/******************************** Tests ***************************************/


TEST_F(SerialBlasTest, Gemm) {

  Interval<float> out[4];

  acopy(4, b, out);
  gemm(1, 1, 1, -2.5, 3.5, c, a, out);
  EXPECT_FINTERVAL_EQ(out[0], -2.5*c[0]*a[0]+3.5*b[0]);

  acopy(4, b, out);
  gemm(2, 2, 1, -1.0, -5.5, c, b, out);
  EXPECT_FINTERVAL_EQ(out[0], Interval<float>(0.0));
  EXPECT_FINTERVAL_EQ(out[1], Interval<float>(-10.5,10.5));
  EXPECT_FINTERVAL_EQ(out[2], Interval<float>(-11.0,11.0));
  EXPECT_FINTERVAL_EQ(out[3], Interval<float>(-20.5,20.5));

  gemm(3, 1, 3, 2.0, 3.0, c, a, b);
  EXPECT_FINTERVAL_EQ(b[0], Interval<float>(-20.0, 28.0));
  EXPECT_FINTERVAL_EQ(b[1], Interval<float>(-5.0, 163.0));
  EXPECT_FINTERVAL_EQ(b[2], Interval<float>(10.0, 406.0));

  Interval<double> x[10000], y[10000], z[10000], w[10000];
  acopy(10000, r->x, x);
  acopy(10000, r->y, y);
  acopy(10000, r->x, z);
  acopy(34*21, r->x, w);

  gemm(34, 21, 89, 4.0, -3.5, x, y, z);

	for (int i = 0; i < 34; ++i) 
		for (int j = 0; j < 21; ++j) {
			Interval<double> sum = 0.0;
			for (int k = 0; k < 89; ++k)
				sum += x[i*89+k]*y[k*21+j];
			w[i*21+j] = -3.5*w[i*21+j] + 4.0*sum;
		}

  for(int i = 0; i < 34*21; ++i)
    EXPECT_DINTERVAL_EQ(z[i], w[i]);
};


TEST_F(SerialBlasTest, Symm) {

  Interval<double> x[10000], y[10000], z[10000], w[10000];
  acopy(10000, r->x, x);
  acopy(10000, r->y, y);
  acopy(10000, r->x, z);
  acopy(10000, r->x, w);

  // Left Upper 
  for(int i = 0; i < 34; ++i)    
		for (int j = 0; j < i; ++j)
      x[i*34+j] = x[j*34+i];
      
  gemm(34, 21, 34, 4.0, -3.5, x, y, z);
  symm('l', 'u', 34, 21, 4.0, -3.5, x, y, w);

  for(int i = 0; i < 34*21; ++i) {
    EXPECT_NEAR(z[i].inf(), w[i].inf(), 1e-6);
    EXPECT_NEAR(z[i].sup(), w[i].sup(), 1e-6);
  }

  // Left Lower
  for(int i = 0; i < 33; ++i)    
	  for (int j = 0; j < i; ++j)
      x[i*33+j] = x[j*33+i];
      
  intlag::gemm(33, 21, 33, 4.0, -3.5, x, y, z);
  symm('L', 'l', 33, 21, 4.0, -3.5, x, y, w);

  for(int i = 0; i < 33*21; ++i) {
    EXPECT_NEAR(z[i].inf(), w[i].inf(), 1e-6);
    EXPECT_NEAR(z[i].sup(), w[i].sup(), 1e-6);
  }

  // Right Upper 
  for(int i = 0; i < 33; ++i)    
		for (int j = 0; j < i; ++j)
      x[i*33+j] = x[j*33+i];
      
  intlag::gemm(20, 33, 33, 4.0, -3.5, y, x, z);
  symm('r', 'U', 20, 33, 4.0, -3.5, x, y, w);

  for(int i = 0; i < 33*20; ++i) {
    EXPECT_NEAR(z[i].inf(), w[i].inf(), 1e-6);
    EXPECT_NEAR(z[i].sup(), w[i].sup(), 1e-6);
  }

  // Right Lower 
  for(int i = 0; i < 33; ++i)    
		for (int j = 0; j < i; ++j)
      x[i*33+j] = x[j*33+i];
      
  intlag::gemm(20, 33, 33, 4.0, -3.5, y, x, z);
  symm('R', 'L', 20, 33, 4.0, -3.5, x, y, w);

  for(int i = 0; i < 33*20; ++i) {
    EXPECT_NEAR(z[i].inf(), w[i].inf(), 1e-6);
    EXPECT_NEAR(z[i].sup(), w[i].sup(), 1e-6);
  }
}


TEST_F(SerialBlasTest, Syrk) {

  Interval<float> x[10000], xtrans[10000], z[10000], w[10000];
  acopy(10000, r->x, x);
  acopy(10000, r->x, xtrans);
  acopy(10000, r->y, z);
  acopy(10000, r->y, w);
  trans(2, 1, xtrans);

  // Upper 
  for(int i = 0; i < 2; ++i)    
		for (int j = 0; j < i; ++j) {
      z[i*2+j] = z[j*2+i];
      w[i*2+j] = w[j*2+i];
    }

  syrk('u', 2, 1, 4.0, -3.5, x, w);
  EXPECT_FINTERVAL_EQ(w[0], 4.0*x[0]*x[0]-3.5*z[0]);
  EXPECT_FINTERVAL_EQ(w[1], 4.0*x[0]*x[1]-3.5*z[1]);
  EXPECT_FINTERVAL_EQ(w[3], 4.0*x[1]*x[1]-3.5*z[3]);

  // Lower
  acopy(10000, r->y, z);
  acopy(10000, r->y, w);
  for(int i = 0; i < 2; ++i)    
		for (int j = 0; j < i; ++j) {
      z[i*2+j] = z[j*2+i];
      w[i*2+j] = w[j*2+i];
    }

  syrk('l', 2, 1, 4.0, -3.5, x, w);
  EXPECT_FINTERVAL_EQ(w[0], 4.0*x[0]*x[0]-3.5*z[0]);
  EXPECT_FINTERVAL_EQ(w[2], 4.0*x[0]*x[1]-3.5*z[2]);
  EXPECT_FINTERVAL_EQ(w[3], 4.0*x[1]*x[1]-3.5*z[3]);
}


TEST_F(SerialBlasTest, Syr2k) {

  Interval<float> x[10000], y[10000], z[10000], w[10000];
  acopy(10000, r->x, x);
  acopy(10000, r->y, y);
  acopy(10000, r->y, z);
  acopy(10000, r->y, w);

  // Upper 
  for(int i = 0; i < 2; ++i)    
		for (int j = 0; j < i; ++j) {
      z[i*2+j] = z[j*2+i];
      w[i*2+j] = w[j*2+i];
    }

  syr2k('u', 2, 1, 4.0, -3.5, x, y, w);
  EXPECT_FINTERVAL_EQ(w[0], 4.0*(x[0]*y[0]+y[0]*x[0])-3.5*z[0]);
  EXPECT_FINTERVAL_EQ(w[1], 4.0*(x[0]*y[1]+y[0]*x[1])-3.5*z[1]);
  EXPECT_FINTERVAL_EQ(w[3], 4.0*(x[1]*y[1]+y[1]*x[1])-3.5*z[3]);

  // Lower
  acopy(10000, r->y, z);
  acopy(10000, r->y, w);
  for(int i = 0; i < 2; ++i)    
		for (int j = 0; j < i; ++j) {
      z[i*2+j] = z[j*2+i];
      w[i*2+j] = w[j*2+i];
    }
      
  syr2k('l', 2, 1, 4.0, -3.5, x, y, w);
  EXPECT_FINTERVAL_EQ(w[0], 4.0*(x[0]*y[0]+y[0]*x[0])-3.5*z[0]);
  EXPECT_FINTERVAL_EQ(w[2], 4.0*(x[0]*y[1]+y[0]*x[1])-3.5*z[2]);
  EXPECT_FINTERVAL_EQ(w[3], 4.0*(x[1]*y[1]+y[1]*x[1])-3.5*z[3]);

}


TEST_F(SerialBlasTest, Trmm) {

  Interval<float> x1[] = {1,2,0,3};
  Interval<float> y1[] = {11,22};
  Interval<float> zeroes1[] = {0,0};
  intlag::gemm(2, 1, 2, 4.0, 0.0, x1, y1, zeroes1);
  trmm('l', 'u', 2, 1, 4.0, x1, y1);
  EXPECT_FINTERVAL_EQ(y1[0], zeroes1[0]);
  EXPECT_FINTERVAL_EQ(y1[1], zeroes1[1]);


  Interval<float> x2[10000], y2[10000], z2[10000], zeroes2[10000];
  acopy(10000, r->x, x2);
  acopy(10000, r->y, y2);
  acopy(10000, r->y, z2);
  acopy(10000, r->y, zeroes2);

  // Left Upper 
  for(int i = 0; i < 34; ++i)    
		for (int j = 0; j < i; ++j)
      x2[i*34+j] = 0.0;
      
  gemm(34, 21, 34, 4.0, 0.0, x2, y2, zeroes2);
  trmm('l', 'u', 34, 21, 4.0, x2, z2);

  for (int i = 0; i < 34; ++i)
    for (int j = 0; j < 21; ++j)
      EXPECT_FINTERVAL_EQ(z2[i*21+j], zeroes2[i*21+j]);


  // Left Lower
  acopy(10000, r->x, x2);
  acopy(10000, r->y, y2);
  acopy(10000, r->y, z2);
  for(int i = 0; i < 33; ++i)    
	  for (int j = i+1; j < 33; ++j)
      x2[i*33+j] = 0.0;

  gemm(33, 21, 33, 4.0, 0.0, x2, y2, zeroes2);
  trmm('L', 'l', 33, 21, 4.0, x2, z2);

  for(int i = 0; i < 33; ++i)    
		for (int j = 0; j < 21; ++j)
      EXPECT_FINTERVAL_EQ(z2[i*21+j], zeroes2[i*21+j]);

  // Right Upper 
  acopy(10000, r->x, x2);
  acopy(10000, r->y, y2);
  acopy(10000, r->y, z2);
  for(int i = 0; i < 20; ++i)    
	  for (int j = 0; j < i; ++j)
      x2[i*20+j] = 0.0;

  gemm(33, 20, 20, 4.0, 0.0, y2, x2, zeroes2);
  trmm('R', 'u', 33, 20, 4.0, x2, z2);

  for(int i = 0; i < 33; ++i)    
		for (int j = 0; j < 20; ++j)
      EXPECT_FINTERVAL_EQ(z2[i*20+j], zeroes2[i*20+j]);

  // Right Lower 
  acopy(10000, r->x, x2);
  acopy(10000, r->y, y2);
  acopy(10000, r->y, z2);
  for(int i = 0; i < 20; ++i)    
	  for (int j = i+1; j < 20; ++j)
      x2[i*20+j] = 0.0;

  gemm(33, 20, 20, 4.0, 0.0, y2, x2, zeroes2);
  trmm('R', 'L', 33, 20, 4.0, x2, z2);

  for(int i = 0; i < 33; ++i)    
		for (int j = 0; j < 20; ++j)
      EXPECT_FINTERVAL_EQ(z2[i*20+j], zeroes2[i*20+j]);
}


TEST_F(SerialBlasTest, Trsm) {

  Interval<double> x1[10000], y1[10000], z1[10000];
  acopy(10000, r->x, x1);
  acopy(10000, r->y, y1);
  acopy(10000, r->y, z1);


  // Left Upper

  // positive diagonal needed
  for(int i = 0; i < 32; ++i)
      x1[i*32+i] = Interval<double>(i+1.0, i+20.0);

  //operation
  scal(32, Interval<double>(-2.7, 4.0), y1);
  trsv('U', 32, x1, y1);
  trsm('L', 'U', 32, 1, Interval<double>(-2.7, 4.0), x1, z1);

  for(int i = 0; i < 32; ++i) {
    EXPECT_NEAR(z1[i].inf(), y1[i].inf(), 1e-6);
    EXPECT_NEAR(z1[i].sup(), y1[i].sup(), 1e-6);
  }

  // Left Lower

  // positive diagonal needed
  for(int i = 0; i < 33; ++i)
      x1[i*33+i] = Interval<double>(i+0.1, i+40.0);

  //operation
  scal(33, Interval<double>(2.7, 4.0), y1);
  trsv('l', 33, x1, y1);
  trsm('L', 'l', 33, 1, Interval<double>(2.7, 4.0), x1, z1);

  for(int i = 0; i < 33; ++i) {
    EXPECT_NEAR(z1[i].inf(), y1[i].inf(), 1e-6);
    EXPECT_NEAR(z1[i].sup(), y1[i].sup(), 1e-6);
  }

  // Right Upper

  //operation
  acopy(10000, r->x, x1);
  acopy(10000, r->y, y1);
  acopy(10000, r->y, z1);
  for(int i = 0; i < 2; ++i)
    x1[i*2+i] = Interval<double>(i+0.1, i+4.0);
  trsm('r', 'U', 1, 2, -5.0, x1, z1);
  EXPECT_DINTERVAL_EQ(z1[0], -5.0*y1[0]/x1[0]);
  EXPECT_DINTERVAL_EQ(z1[1], (-5.0*y1[1]-z1[0]*x1[1])/x1[3]);

  // Right Lower

  //operation
  acopy(10000, r->x, x1);
  acopy(10000, r->y, y1);
  acopy(10000, r->y, z1);
  for(int i = 0; i < 2; ++i)
    x1[i*2+i] = Interval<double>(i+0.1, i+4.0);
  trsm('r', 'L', 1, 2, -5.0, x1, z1);
  EXPECT_DINTERVAL_EQ(z1[1], -5.0*y1[1]/x1[3]);
  EXPECT_DINTERVAL_EQ(z1[0], (-5.0*y1[0]-z1[1]*x1[2])/x1[0]);
}


#endif



