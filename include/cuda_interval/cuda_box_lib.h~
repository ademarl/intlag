
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_BOX_LIB_H
#define CUDA_BOX_LIB_H

#include <assert.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "cuda_interval_lib.h"

#include "cuda_box.h"
#include "blas/cuda_blas.h"
#include "aux/device_data.h"


namespace intlag {

//Unary Sign Operators
template<class T> inline __host__
CudaBox<T> operator+(CudaBox<T> const &x) {
  return CudaBox<T>(x);
}


template<class T> inline __host__
CudaBox<T> operator-(CudaBox<T> const &x) {
  CudaBox<T> ret(x);
  for(int i = 0; i < x.length(); ++i)
    ret.elements[i] = -ret[i];
  return ret;
}


// CudaBox sum
template<class T, class F> inline __host__
CudaBox<T> operator+(CudaBox<T> const &x, CudaBox<F> const &y) {
  unsigned int n = x.length();
  if (n != y.length())
    throw out_of_range("CudaBox dimensions are not equal in sum.");

  CudaBox<T> ret(n);

	DeviceData< CudaInterval<T> > d_x(n, x.data()), d_y(n, y.data());

	CudaGeneral::axpy(n, 1.0, d_x.data(), d_y.data());

	d_y.toHost(ret.data());

  return ret;
}

// CudaBox sum
template<class T> inline __host__
CudaBox<T> operator-(CudaBox<T> const &x, CudaBox<T> const &y) {
	return CudaBox<T>(x+(-y));
}

// CudaBox scalar multiplication of form alpha*x
template<class T, class F> inline __host__
CudaBox<T> operator*(F alpha, CudaBox<T> const &x) {
  int n = x.length();
  CudaBox<T> ret(n);

	DeviceData< CudaInterval<T> > d_x(n, x.data());

  CudaGeneral::scal(n, alpha, d_x.data());

	d_x.toHost(ret.data());

	return ret;
}


// CudaBox scalar multiplication of form x*alpha
template<class T, class F> inline __host__
CudaBox<T> operator*(CudaBox<T> const &x, F alpha) {
	return alpha*x;
}


// CudaBox scalar multiplication of form alpha*x
template<class T, class F> inline __host__
CudaBox<T> operator*(CudaBox<T> const &x, CudaBox<F> const &y) {
  unsigned int n1 = x.lins();
  unsigned int m1 = x.cols();
  unsigned int n2 = y.lins();
  unsigned int m2 = y.cols();
  if (m1 != n2)
    throw out_of_range("CudaBox dimensions are not valid on multiplication.");

  CudaBox<T> ret(n1, m2);

	DeviceData< CudaInterval<T> > d_x(n1*m1, x.data()), d_y(n2*m2, y.data()), d_z(n1*m2);

  if(m2 == 1)
    gemv(n1, m1, 1.0, 0.0, x.data(), y.data(), d_z.data());
  else
    gemm(n1, m2, m1, 1.0, 0.0, x.data(), y.data(), d_z.data());

	d_z.toHost(ret.data());

	return ret;
}


} // namespace intlag

#endif



