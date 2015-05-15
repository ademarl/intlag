
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef TEST_INTERVAL_H
#define TEST_INTERVAL_H


#include "gtest/gtest.h"

template <class T, class U>
void EXPECT_FINTERVAL_EQ(T x, U y) {
  EXPECT_FLOAT_EQ(x.inf(), y.inf());
  EXPECT_FLOAT_EQ(x.sup(), y.sup());
}

template <class T, class U>
void EXPECT_DINTERVAL_EQ(T x, U y) {
  EXPECT_DOUBLE_EQ(x.inf(), y.inf());
  EXPECT_DOUBLE_EQ(x.sup(), y.sup());
}

template <class T, class U>
void EXPECT_FINTERVAL_NEAR(T x, U y, double eps) {
  EXPECT_NEAR(x.inf(), y.inf(), eps);
  EXPECT_NEAR(x.sup(), y.sup(), eps);
}

template <class T, class U>
void EXPECT_DINTERVAL_NEAR(T x, U y, double eps) {
  EXPECT_NEAR(x.inf(), y.inf(), eps);
  EXPECT_NEAR(x.sup(), y.sup(), eps);
}

#endif
