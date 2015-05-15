
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_INTERVAL_LIB_H
#define CUDA_INTERVAL_LIB_H

#include "aux/min_max.h"
//#include "interval/interval_lib.h"
#include "cuda_interval.h"


namespace intlag {


template <class T> inline __device__ __host__
bool contain_zero(CudaInterval<T> const x)
{
  if (x.inf() <= 0 && x.sup() >= 0) return true;
  return false;
}


template <class T> inline __device__ __host__
bool empty(CudaInterval<T> const x)
{
  if (isnan(x.inf()) || isnan(x.sup())) return true;
  return false;
}

// add host functionality (test)
template <class T> inline __device__ __host__
CudaInterval<T> abs(CudaInterval<T> const x)
{
  return CudaInterval<T>( max(x.inf(), -x.sup())*((int)!contain_zero(x)), max(-x.inf(), x.sup()) );;
}


template<class T> inline __device__ __host__
T width(CudaInterval<T> x)
{
  if (empty(x))
      return 0;

  return CudaRounder<T>::sub_up(x.inf(), x.sup());
}


// Median - rounded to nearest
template <class T> inline __device__
T median(CudaInterval<T> const x)
{
  if (empty(x))
    return CudaRounder<T>::nan();
  return CudaRounder<T>::div_up(width(x), 2.0) + x.inf();
}


// Median - rounded up - put this in Rounder
template <class T> inline __device__
T median_up(CudaInterval<T> const x)
{
  if (empty(x))
    return CudaRounder<T>::nan();
  return CudaRounder<T>::add_up(CudaRounder<T>::div_up(width(x), 2.0), x.inf());
}


// Median - rounded down - put this in Rounder
template <class T> inline __device__
T median_down(CudaInterval<T> const x)
{
  if (empty(x))
    return CudaRounder<T>::nan();
  return CudaRounder<T>::add_down(CudaRounder<T>::div_up(width(x), 2.0), x.inf());
}


// Intersection
template <class T> inline __device__
CudaInterval<T> overlap(CudaInterval<T> const x, CudaInterval<T> const y)
{
  if(empty(x) || empty(y)) return CudaInterval<T>::empty();

  T u = max(x.inf(), y.inf());
  T v = min(x.sup(), y.sup());

  if (u <= v)
    return CudaInterval<T>(u, v);
  return CudaInterval<T>::empty();
}


// Intersection
template <class T> inline __device__
CudaInterval<T> hull(CudaInterval<T> const x, CudaInterval<T> const y)
{
  if(empty(x) || empty(y)) return CudaInterval<T>::empty();

  T u = min(x.inf(), y.inf());
  T v = max(x.sup(), y.sup());

  if (u <= v)
    return CudaInterval<T>(u, v);
  return CudaInterval<T>::empty();
}


// Unary
template <class T> inline __device__ __host__
CudaInterval<T> operator+(CudaInterval<T> const x)
{
    return x;
}


template <class T> inline __device__ __host__
CudaInterval<T> operator-(CudaInterval<T> const x)
{
    return CudaInterval<T>(-x.sup(), -x.inf());
}


// Binary
template <class T> inline __device__
CudaInterval<T> operator+(CudaInterval<T> const x, CudaInterval<T> const y)
{
    return CudaInterval<T>( CudaRounder<T>::add_down(x.inf(), y.inf()),
                            CudaRounder<T>::add_up(x.sup(), y.sup())   );
}

template <class T> inline __device__
CudaInterval<T> operator-(CudaInterval<T> const x, CudaInterval<T> const y)
{
    return CudaInterval<T>( CudaRounder<T>::sub_down(x.inf(), y.sup()),
                            CudaRounder<T>::sub_up(x.sup(), y.inf())   );
}


template <class T, class F> inline __device__
CudaInterval<T> operator*(F const a, CudaInterval<T> const x)
{
    return CudaInterval<T>(  min( CudaRounder<T>::mul_down(a, x.inf()),
                                  CudaRounder<T>::mul_down(a, x.sup()) ),
                             max( CudaRounder<T>::mul_up(a, x.inf()),
                                  CudaRounder<T>::mul_up(a, x.sup())   ));
}


template <class T, class F> inline __device__
CudaInterval<T> operator*(CudaInterval<T> const x, F const a)
{
    return CudaInterval<T>(  min( CudaRounder<T>::mul_down(a, x.inf()),
                                  CudaRounder<T>::mul_down(a, x.sup()) ),
                             max( CudaRounder<T>::mul_up(a, x.inf()),
                                  CudaRounder<T>::mul_up(a, x.sup())   ));
}


template <class T> inline __device__
CudaInterval<T> operator*(CudaInterval<T> const x, CudaInterval<T> const y)
{
  return CudaInterval<T>(  min( CudaRounder<T>::mul_down(x.inf(), y.inf()),
                                CudaRounder<T>::mul_down(x.inf(), y.sup()),
                                CudaRounder<T>::mul_down(x.sup(), y.inf()),
                                CudaRounder<T>::mul_down(x.sup(), y.sup()) ),
                           max( CudaRounder<T>::mul_up(x.inf(), y.inf()),
                                CudaRounder<T>::mul_up(x.inf(), y.sup()),
                                CudaRounder<T>::mul_up(x.sup(), y.inf()),
                                CudaRounder<T>::mul_up(x.sup(), y.sup())) );
}


template <class T> inline __device__
CudaInterval<T> operator/(CudaInterval<T> const x, CudaInterval<T> const y)
{
  if(contain_zero(y)) return CudaInterval<T>::empty();

  return CudaInterval<T>(  min( CudaRounder<T>::div_down(x.inf(), y.inf()),
                                CudaRounder<T>::div_down(x.inf(), y.sup()),
                                CudaRounder<T>::div_down(x.sup(), y.inf()),
                                CudaRounder<T>::div_down(x.sup(), y.sup()) ),
                           max( CudaRounder<T>::div_up(x.inf(), y.inf()),
                                CudaRounder<T>::div_up(x.inf(), y.sup()),
                                CudaRounder<T>::div_up(x.sup(), y.inf()),
                                CudaRounder<T>::div_up(x.sup(), y.sup())) );
}


// Other methods
template <class T, class F> inline __device__
CudaInterval<T> fma(F const a, CudaInterval<T> const x, CudaInterval<T> const y)
{
    return CudaInterval<T>(  min( CudaRounder<T>::fma_down(a, x.inf(), y.inf()),
                                  CudaRounder<T>::fma_down(a, x.sup(), y.inf()) ),
                             max( CudaRounder<T>::fma_up(a, x.inf(), y.sup()),
                                  CudaRounder<T>::fma_up(a, x.sup(), y.sup())   ));
}


template <class T, class F> inline __device__
CudaInterval<T> fma(CudaInterval<F> const a, CudaInterval<T> const x, CudaInterval<T> const y)
{
    return CudaInterval<T>(  min( CudaRounder<T>::fma_down(a.inf(), x.inf(), y.inf()),
                                  CudaRounder<T>::fma_down(a.inf(), x.sup(), y.inf()),
                                  CudaRounder<T>::fma_down(a.sup(), x.inf(), y.inf()),
                                  CudaRounder<T>::fma_down(a.sup(), x.sup(), y.inf()) ),
                             max( CudaRounder<T>::fma_up(a.inf(), x.inf(), y.sup()),
                                  CudaRounder<T>::fma_up(a.inf(), x.sup(), y.sup()),
                                  CudaRounder<T>::fma_up(a.sup(), x.inf(), y.sup()),
                                  CudaRounder<T>::fma_up(a.sup(), x.sup(), y.sup())   ));
}


template <class T> inline __device__
CudaInterval<T> sqrt(CudaInterval<T> const x)
{
	return CudaInterval<T>(	CudaRounder<T>::sqrt_down( x.inf() ), CudaRounder<T>::sqrt_up( x.sup() ) );
}


} // namespace ilag

#endif



