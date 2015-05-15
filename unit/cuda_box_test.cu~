
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef CUDA_BOX_TEST_CU
#define CUDA_BOX_TEST_CU

#include <math.h>
#include "cuda_interval/cuda_box_lib.h"
#include "gtest/gtest.h"

#include "aux/reference.h"
#include "aux/test_interval.h"
#include "blas/cuda_blas.h"

using namespace intlag;


class CudaCudaBoxTest : public ::testing::Test {
  protected:

    CudaCudaBoxTest() {
      ref = Reference::getInstance();
    }
    virtual ~CudaCudaBoxTest() {}

    virtual void SetUp() {

      for(int i = 0; i < 300; ++i) {
        acopy(300, ref->x, a);
        acopy(300, ref->y, b);
        acopy(300, ref->x, c);
        acopy(300, ref->x, da);
        acopy(300, ref->y, db);
        acopy(300, ref->x, dc);
      }

    }
    virtual void TearDown() {}

    Reference *ref;

    CudaInterval<float> a[300], b[300], c[300];
    CudaInterval<double> da[300], db[300], dc[300];
};




/******************************** Tests ***************************************/


// Tests constructors and access
TEST_F(CudaCudaBoxTest, Constructors) {

  CudaBox< double > bx;
  EXPECT_EQ(bx.lins(), 0);
  EXPECT_EQ(bx.cols(), 0);
  EXPECT_EQ(bx.length(), 0);
  EXPECT_TRUE(bx.empty());

  CudaBox< double > by(5);
  EXPECT_EQ(by.lins(), 5);
  EXPECT_EQ(by.cols(), 1);
  EXPECT_EQ(by.length(), 5);
  EXPECT_FALSE(by.empty());

  CudaBox< double > bz(3, 5);
  EXPECT_EQ(bz.lins(), 3);
  EXPECT_EQ(bz.cols(), 5);
  EXPECT_EQ(bz.length(), 15);
  EXPECT_FALSE(bz.empty());

  CudaBox< double > ba(3, da);
  EXPECT_EQ(ba.lins(), 3);
  EXPECT_EQ(ba.cols(), 1);
  EXPECT_EQ(ba.length(), 3);
  for(int i = 0; i < 3; ++i)
    EXPECT_DINTERVAL_EQ(ba[i], da[i]);

  CudaBox< double > bb(4, 8, db);
  EXPECT_EQ(bb.lins(), 4);
  EXPECT_EQ(bb.cols(), 8);
  EXPECT_EQ(bb.length(), 32);
  for(int i = 0; i < 32; ++i)
    EXPECT_DINTERVAL_EQ(bb[i], db[i]);

  CudaBox< double > bc(bb);
  EXPECT_EQ(bc.lins(), 4);
  EXPECT_EQ(bc.cols(), 8);
  EXPECT_EQ(bc.length(), 32);
  for(int i = 0; i < 32; ++i)
    EXPECT_DINTERVAL_EQ(bc[i], bb.at(i));
};


TEST_F(CudaCudaBoxTest, Comparison) {

  EXPECT_TRUE(CudaBox<float>(3,a) == CudaBox<float>(3,1,a));
  EXPECT_FALSE(CudaBox<float>(3,a) == CudaBox<float>(3,1,b));
  EXPECT_FALSE(CudaBox<float>(3,a) == CudaBox<float>(5,1,a));
  EXPECT_TRUE(CudaBox<float>(3,a) == a);
  EXPECT_TRUE(a == CudaBox<float>(3,a));
  EXPECT_FALSE(CudaBox<float>(3,a) != a);
  EXPECT_FALSE(a != CudaBox<float>(3,a));
  EXPECT_TRUE(CudaBox<float>(3,a) != b);
  EXPECT_TRUE(a != CudaBox<float>(3,b));
}

TEST_F(CudaCudaBoxTest, Assignment) {

  CudaBox< double > bx(2, 5);
  bx = da;
  EXPECT_EQ(bx.lins(), 2);
  EXPECT_EQ(bx.cols(), 5);
  EXPECT_EQ(bx.length(), 10);
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(bx[i], da[i]);

  CudaBox< double > by = bx;
  EXPECT_EQ(by.lins(), 2);
  EXPECT_EQ(by.cols(), 5);
  EXPECT_EQ(by.length(), 10);
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(by[i], da[i]);
};

TEST_F(CudaCudaBoxTest, Unary) {

  CudaBox< double > bx(2, 5, da), by;

  by = +bx;
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(by[i], da[i]);

  by = -bx;
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(by[i], -da[i]);
}




TEST_F(CudaCudaBoxTest, Sum) {

  CudaBox< double > bx(80, 3, da), by(80, 3, db), bz;

  bz = bx+by;
  CudaGeneralManaged::axpy(240, 1.0, da, db);
  for(int i = 0; i < 240; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], db[i]);
}


TEST_F(CudaCudaBoxTest, Sub) {

  CudaBox< double > bx(80, 3, da), by(80, 3, db), bz;

  bz = bx-by;
  CudaGeneralManaged::axpy(240, -1.0, db, da);
  for(int i = 0; i < 240; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], da[i]);
}


TEST_F(CudaCudaBoxTest, ScalarMult) {

  CudaBox< double > bx(75, 4, da), bz;
  bz = bx*(-3.2);

  CudaGeneralManaged::scal(75*4, -3.2, da);

  for(int i = 0; i < 300; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], da[i]);
}


TEST_F(CudaCudaBoxTest, Gemv) {

  CudaBox< double > bx(5, 44, da), by(44, db), bz;

  bz = bx*by;
  CudaGeneralManaged::gemv(5, 44, 1.0, 0.0, da, db, dc);

  for(int i = 0; i < 5; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], dc[i]);
}


TEST_F(CudaCudaBoxTest, Gemm) {

  CudaBox< double > bx(44, 5, da), by(5, 6, db), bz;

  bz = bx*by;
  gemm(44, 6, 5, 1.0, 0.0, da, db, dc);

  for(int i = 0; i < 44*6; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], dc[i]);
}


#endif



