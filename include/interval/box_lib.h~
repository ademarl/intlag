
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef BOX_LIB_H
#define BOX_LIB_H

#include <assert.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "interval_lib.h"

#include "box.h"
#include "../blas/cuda_blas.h"
#include "../blas/serial_blas.h"


namespace intlag {

//Unary Signs
template<class T> inline __host__
Box<T> operator+(Box<T> const &x) {
  return Box<T>(x);
}


template<class T> inline __host__
Box<T> operator-(Box<T> const &x) {
  Box<T> ret(x);
  for(int i = 0; i < x.length(); ++i)
    ret.elements[i] = -ret[i];
  return ret;
}


// Box sum
template<class T, class F> inline __host__
Box<T> operator+(Box<T> const &x, Box<F> const &y) {
  unsigned int n = x.length();

  if (n != y.length())
    throw out_of_range("Box dimensions are not equal in sum.");

  Box<T> ret = y;

  axpy(n, 1.0, x.data(), ret.data());

  return ret;
}

// Box sum
template<class T> inline __host__
Box<T> operator-(Box<T> const &x, Box<T> const &y) {
	return Box<T>(x+(-y));
}

// Box scalar multiplication of form alpha*x
template<class T, class F> inline __host__
Box<T> operator*(F alpha, Box<T> const &x) {

  Box<T> ret = x;

  scal(x.length(), alpha, ret.data());

	return ret;
}


// Box scalar multiplication of form x*alpha
template<class T, class F> inline __host__
Box<T> operator*(Box<T> const &x, F alpha) {
	return alpha*x;
}


// Box scalar multiplication of form alpha*x
template<class T, class F> inline __host__
Box<T> operator*(Box<T> const &x, Box<F> const &y) {
  unsigned int n1 = x.lins();
  unsigned int m1 = x.cols();
  unsigned int n2 = y.lins();
  unsigned int m2 = y.cols();
  if (m1 != n2)
    throw out_of_range("Box dimensions are not valid on multiplication.");

  Box<T> ret(n1, m2, Interval<T>(0.0));

  if(m2 == 1)
    gemv(n1, m1, 1.0, 0.0, x.data(), y.data(), ret.data());
  else
    gemm(n1, m2, m1, 1.0, 0.0, x.data(), y.data(), ret.data());

	return ret;
}


/*template<class T, class F> inline __host__
Box<T> operator+(Box<T> const &x, Box<F> const &y) {
  unsigned int n = x.length();
  if (n != y.length())
    throw out_of_range("Box dimensions are not equal in sum.");

  DevideData< T > dx(n, x.data()), dy(n, y.data());
  CudaGeneral::axpy(1.0, dx.data(), dy.data());

  Box<T> ret(x.lins, x.cols);
  dy.toHost(ret.data());

  return ret;
}*/

} // namespace intlag

#endif



