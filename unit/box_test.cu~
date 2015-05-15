
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef BOX_TEST_CU
#define BOX_TEST_CU

#include <math.h>
#include "interval/box_lib.h"
#include "gtest/gtest.h"

#include "aux/reference.h"
#include "aux/test_interval.h"
#include "blas/serial_blas.h"

using namespace intlag;


class BoxTest : public ::testing::Test {
  protected:

    BoxTest() {
      ref = Reference::getInstance();
    }
    virtual ~BoxTest() {}

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

    Interval<float> a[300], b[300], c[300];
    Interval<double> da[300], db[300], dc[300];
};




/******************************** Tests ***************************************/


// Tests constructors and access
TEST_F(BoxTest, Constructors) {

  Box< double > bx;
  EXPECT_EQ(bx.lins(), 0);
  EXPECT_EQ(bx.cols(), 0);
  EXPECT_EQ(bx.length(), 0);
  EXPECT_TRUE(bx.empty());

  Box< double > by(5);
  EXPECT_EQ(by.lins(), 5);
  EXPECT_EQ(by.cols(), 1);
  EXPECT_EQ(by.length(), 5);
  EXPECT_FALSE(by.empty());

  Box< double > bz(3, 5);
  EXPECT_EQ(bz.lins(), 3);
  EXPECT_EQ(bz.cols(), 5);
  EXPECT_EQ(bz.length(), 15);
  EXPECT_FALSE(bz.empty());

  Box< double > ba(3, da);
  EXPECT_EQ(ba.lins(), 3);
  EXPECT_EQ(ba.cols(), 1);
  EXPECT_EQ(ba.length(), 3);
  for(int i = 0; i < 3; ++i)
    EXPECT_DINTERVAL_EQ(ba[i], da[i]);

  Box< double > bb(4, 8, db);
  EXPECT_EQ(bb.lins(), 4);
  EXPECT_EQ(bb.cols(), 8);
  EXPECT_EQ(bb.length(), 32);
  for(int i = 0; i < 32; ++i)
    EXPECT_DINTERVAL_EQ(bb[i], db[i]);

  Box< double > bc(bb);
  EXPECT_EQ(bc.lins(), 4);
  EXPECT_EQ(bc.cols(), 8);
  EXPECT_EQ(bc.length(), 32);
  for(int i = 0; i < 32; ++i)
    EXPECT_DINTERVAL_EQ(bc[i], bb.at(i));
};


TEST_F(BoxTest, Comparison) {

  EXPECT_TRUE(Box<float>(3,a) == Box<float>(3,1,a));
  EXPECT_FALSE(Box<float>(3,a) == Box<float>(3,1,b));
  EXPECT_FALSE(Box<float>(3,a) == Box<float>(5,1,a));
  EXPECT_TRUE(Box<float>(3,a) == a);
  EXPECT_TRUE(a == Box<float>(3,a));
  EXPECT_FALSE(Box<float>(3,a) != a);
  EXPECT_FALSE(a != Box<float>(3,a));
  EXPECT_TRUE(Box<float>(3,a) != b);
  EXPECT_TRUE(a != Box<float>(3,b));
}

TEST_F(BoxTest, Assignment) {

  Box< double > bx(2, 5);
  bx = da;
  EXPECT_EQ(bx.lins(), 2);
  EXPECT_EQ(bx.cols(), 5);
  EXPECT_EQ(bx.length(), 10);
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(bx[i], da[i]);

  Box< double > by = bx;
  EXPECT_EQ(by.lins(), 2);
  EXPECT_EQ(by.cols(), 5);
  EXPECT_EQ(by.length(), 10);
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(by[i], da[i]);
};

TEST_F(BoxTest, Unary) {

  Box< double > bx(2, 5, da), by;

  by = +bx;
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(by[i], da[i]);

  by = -bx;
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(by[i], -da[i]);
}




TEST_F(BoxTest, Sum) {

  Box< double > bx(2, 5, da), by(2, 5, db), bz;

  bz = bx+by;
  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], da[i]+db[i]);
}


TEST_F(BoxTest, Sub) {

  Box< double > bx(2, 5, da), by(2, 5, db), bz;

  bz = bx-by;

  for(int i = 0; i < 10; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], da[i]-db[i]);
}


TEST_F(BoxTest, ScalarMult) {

  Box< double > bx(2, 5, da), by, bz;

  by = 3.5*bx;
  bz = bx*(-3.2);

  for(int i = 0; i < 10; ++i) {
    EXPECT_DINTERVAL_EQ(by[i], 3.5*da[i]);
    EXPECT_DINTERVAL_EQ(bz[i], -3.2*da[i]);

  }
}


TEST_F(BoxTest, Gemv) {

  Box< double > bx(2, 5, da), by(5, db), bz;

  bz = bx*by;
  gemv(2, 5, 1.0, 0.0, da, db, dc);

  for(int i = 0; i < 2; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], dc[i]);
}


TEST_F(BoxTest, Gemm) {

  Box< double > bx(2, 5, da), by(5, 8, db), bz;

  bz = bx*by;
  gemm(2, 8, 5, 1.0, 0.0, da, db, dc);

  for(int i = 0; i < 2*8; ++i)
    EXPECT_DINTERVAL_EQ(bz[i], dc[i]);
}


#endif



