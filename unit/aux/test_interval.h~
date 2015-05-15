
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef EXPECT_INTERVAL_H
#define EXPECT_INTERVAL_H


#include "gtest/gtest.h"
#include "cuda_interval/cuda_interval_lib.h"
#include "interval/interval_lib.h"
using namespace intlag;


template <class T, class U>
void EXPECT_FINTERVAL_EQ(Interval<T> x, Interval<U> y) {
  EXPECT_FLOAT_EQ(x.inf(), y.inf());
  EXPECT_FLOAT_EQ(x.sup(), y.sup());
}

template <class T, class U>
void EXPECT_DINTERVAL_EQ(Interval<T> x, Interval<U> y) {
  EXPECT_DOUBLE_EQ(x.inf(), y.inf());
  EXPECT_DOUBLE_EQ(x.sup(), y.sup());
}


template <class T>
void EXPECT_FINTERVAL_EQ(Interval<T> x, float y) {
  EXPECT_FLOAT_EQ(x.inf(), y);
  EXPECT_FLOAT_EQ(x.sup(), y);
}

template <class T>
void EXPECT_DINTERVAL_EQ(Interval<T> x, double y) {
  EXPECT_DOUBLE_EQ(x.inf(), y);
  EXPECT_DOUBLE_EQ(x.sup(), y);
}

template <class T, class U>
void EXPECT_FINTERVAL_EQ(CudaInterval<T> x, CudaInterval<U> y) {
  EXPECT_FLOAT_EQ(x.inf(), y.inf());
  EXPECT_FLOAT_EQ(x.sup(), y.sup());
}

template <class T, class U>
void EXPECT_DINTERVAL_EQ(CudaInterval<T> x, CudaInterval<U> y) {
  EXPECT_DOUBLE_EQ(x.inf(), y.inf());
  EXPECT_DOUBLE_EQ(x.sup(), y.sup());
}

template <class T>
void EXPECT_FINTERVAL_EQ(CudaInterval<T> x, float y) {
  EXPECT_FLOAT_EQ(x.inf(), y);
  EXPECT_FLOAT_EQ(x.sup(), y);
}

template <class T>
void EXPECT_DINTERVAL_EQ(CudaInterval<T> x, double y) {
  EXPECT_DOUBLE_EQ(x.inf(), y);
  EXPECT_DOUBLE_EQ(x.sup(), y);
}

template <class T, class U>
void EXPECT_FINTERVAL_EQ(CudaInterval<T> x, Interval<U> y) {
  EXPECT_FLOAT_EQ(x.inf(), y.inf());
  EXPECT_FLOAT_EQ(x.sup(), y.sup());
}

template <class T, class U>
void EXPECT_DINTERVAL_EQ(CudaInterval<T> x, Interval<U> y) {
  EXPECT_DOUBLE_EQ(x.inf(), y.inf());
  EXPECT_DOUBLE_EQ(x.sup(), y.sup());
}

template <class T, class U>
void EXPECT_FINTERVAL_EQ(Interval<T> x, CudaInterval<U> y) {
  EXPECT_FLOAT_EQ(x.inf(), y.inf());
  EXPECT_FLOAT_EQ(x.sup(), y.sup());
}

template <class T, class U>
void EXPECT_DINTERVAL_EQ(Interval<T> x, CudaInterval<U> y) {
  EXPECT_DOUBLE_EQ(x.inf(), y.inf());
  EXPECT_DOUBLE_EQ(x.sup(), y.sup());
}


#endif
