
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef CUDA_INTERVAL_TEST_CU
#define CUDA_INTERVAL_TEST_CU


#include "aux/reference.h"
#include "cuda_interval/cuda_interval_lib.h"
#include "gtest/gtest.h"
#include "aux/test_interval.h"

using namespace intlag;


class CudaIntervalTest : public ::testing::Test {
  protected:

    CudaIntervalTest() {
      r = Reference::getInstance();
    }
    virtual ~CudaIntervalTest(){}

    virtual void SetUp()
    { 
      neg = CudaInterval<float>(-3.1, -1.2);
      mid = CudaInterval<float>(-5.1, 5.1);
      pos = CudaInterval<float>(2.2, 4.4);

      zero = CudaInterval<float>(0.0);
      point_pos = CudaInterval<float>(2.0);
      point_neg = CudaInterval<float>(-2.0);

      empty_i = CudaInterval<float>();
      empty_up = CudaInterval<float>(nanf(""), 2048.8);
      empty_down = CudaInterval<float>(-3000.8, nanf(""));

      dneg = CudaInterval<double>(-3.1, -1.2);
      dmid = CudaInterval<double>(-5.1, 5.1);
      dpos = CudaInterval<double>(2.2, 4.4);

      dzero = CudaInterval<double>(0.0);
      dpoint_pos = CudaInterval<double>(2.0);
      dpoint_neg = CudaInterval<double>(-2.0);

      dempty_i = CudaInterval<double>();
      dempty_up = CudaInterval<double>(nan(""), 2048.8);
      dempty_down = CudaInterval<double>(-3000.8, nan(""));

      a = b = c = CudaInterval<float>::empty();
      da = db = dc = CudaInterval<double>::empty();
    }

    virtual void TearDown() {}

    Reference *r;

    CudaInterval<float> z, dz;
    CudaInterval<float> a, b, c;
    CudaInterval<double> da, db, dc;

    CudaInterval<float> neg, mid, pos, zero, point_pos, point_neg, empty_i, empty_up, empty_down;
    CudaInterval<double> dneg, dmid, dpos, dzero, dpoint_pos, dpoint_neg, dempty_i, dempty_up, dempty_down;

};

TEST_F(CudaIntervalTest, Constructors) {

  EXPECT_TRUE(isnan( (z = r->empty_i).inf() ));
  EXPECT_TRUE(isnan( (z = r->empty_i).sup() ));
  EXPECT_TRUE(isnan( (new CudaInterval<float>)->inf() ));
  EXPECT_TRUE(isnan( (new CudaInterval<float>)->sup() ));
  EXPECT_TRUE(isnan( (new CudaInterval<double>)->inf() ));
  EXPECT_TRUE(isnan( (new CudaInterval<double>)->sup() ));

  EXPECT_TRUE(isnan( (z = r->empty_up).sup() ));
  EXPECT_TRUE(isnan( (z = r->empty_down).inf() ));

  EXPECT_DEATH(CudaInterval<float>(150.1, 130.3), "");
  EXPECT_DEATH(CudaInterval<float>(2.0, -3.0), "");
  EXPECT_DEATH(CudaInterval<double>(-102.0, -300.5), "");

  EXPECT_FLOAT_EQ( (z = r->point_neg).inf(), -2.0);
  EXPECT_FLOAT_EQ( (z = r->point_pos).sup(), 2.0);
  EXPECT_FLOAT_EQ( (z = r->neg).inf(), -3.1);
  EXPECT_FLOAT_EQ( (z = r->pos).sup(), 4.4);

  EXPECT_DOUBLE_EQ(r->point_pos.inf(), 2.0);
  EXPECT_DOUBLE_EQ(r->point_neg.sup(), -2.0);
  EXPECT_DOUBLE_EQ(r->pos.inf(), 2.2);
  EXPECT_DOUBLE_EQ(r->neg.sup(), -1.2);
};


TEST_F(CudaIntervalTest, ContainZero) {

EXPECT_TRUE(contain_zero(z = r->zero));
EXPECT_TRUE(contain_zero(z = r->mid));
EXPECT_FALSE(contain_zero(z = r->neg));
EXPECT_FALSE(contain_zero(z = r->pos));
}


TEST_F(CudaIntervalTest, Empty) {

  // Method empty creates an empty interval
  EXPECT_TRUE(isnan( CudaInterval<float>::empty().inf() )); // static call
  EXPECT_TRUE(isnan( CudaInterval<double>::empty().sup() ));
  
  // Empty function, checks if an interval is empty
  EXPECT_TRUE(empty(z = r->mid.empty()));
  EXPECT_TRUE(empty(z = r->empty_i));
  EXPECT_TRUE(empty(z = r->empty_up));
  EXPECT_TRUE(empty(z = r->empty_down));
  EXPECT_FALSE(empty(z = r->neg));
  EXPECT_FALSE(empty(z = r->mid));
  EXPECT_FALSE(empty(z = r->pos));
};


TEST_F(CudaIntervalTest, Equality) {

  EXPECT_TRUE(pos != mid);
  EXPECT_TRUE(pos != mid);
  EXPECT_FALSE(dpoint_pos == dpoint_neg);
  EXPECT_EQ(mid, CudaInterval<float>(-5.1, 5.1));
  EXPECT_EQ(dmid, dmid);
  EXPECT_EQ(point_pos, 2.0);
  EXPECT_EQ(-2.0, point_neg);
}


TEST_F(CudaIntervalTest, Attribution) {

  z = 2.0;
  EXPECT_EQ(z, CudaInterval<float>(2.0));
  z = CudaInterval<double>(3.5, 4.2);
  EXPECT_EQ(z, CudaInterval<float>(3.5, 4.2));
}
#endif



