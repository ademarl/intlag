
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CUDA_INTERVAL_H
#define CUDA_INTERVAL_H

#include <assert.h>
#include "../interval/interval_lib.h"
#include "cuda_rounder.h"

namespace intlag {


template <class T>
class CudaInterval {
    public:

        __device__ __host__ CudaInterval();
         __host__ CudaInterval(Interval<T> const x);
        __device__ __host__ CudaInterval(T const x);
        __device__ __host__ CudaInterval(T const x, T const y);

        __device__ __host__ T const &inf() const;
        __device__ __host__ T const &sup() const;

        static __device__ __host__ CudaInterval empty();

        

        template <class F> __host__
        CudaInterval& operator=(const Interval<F> &rhs);

        template <class F> __device__ __host__
        CudaInterval& operator=(const CudaInterval<F> &rhs);

        __device__ __host__ CudaInterval& operator=(const double &rhs);

    private:
        T inf_;
        T sup_;
};



// Constructors
template <class T> inline __device__ __host__
CudaInterval<T>::CudaInterval() :
  inf_(CudaRounder<T>::nan()), sup_(CudaRounder<T>::nan())
{
}


template <class T> inline __host__
CudaInterval<T>::CudaInterval(Interval<T> const x) :
    inf_(x.inf()), sup_(x.sup())
{
}

template <class T> inline __device__ __host__
CudaInterval<T>::CudaInterval(T const x) :
    inf_(x), sup_(x)
{
}


template <class T> inline __device__ __host__
CudaInterval<T>::CudaInterval(T const x, T const y) :
  inf_(x), sup_(y)
{
  
  if (isnan(x) || isnan(y)) return;
  assert(inf_ <= sup_); // handle this error better
}


// Getters
template <class T> inline __device__ __host__
T const &CudaInterval<T>::inf() const
{
    return inf_;
}


template <class T> inline __device__ __host__
T const &CudaInterval<T>::sup() const
{
    return sup_;
}


template <class T> inline __device__ __host__
CudaInterval<T> CudaInterval<T>::empty()
{
    return CudaInterval<T>(CudaRounder<T>::nan(), CudaRounder<T>::nan());
}


// Logical Comparison: FIXME: assimetric comparations, rounding when casting?
template <class T, class F> inline __device__ __host__
bool operator==(const CudaInterval<T>& lhs, const CudaInterval<F>& rhs)
{
  return lhs.inf() == (T) rhs.inf() && lhs.sup() == (T) rhs.sup();
}


template <class T, class F> inline __host__
bool operator==(const CudaInterval<T>& lhs, const Interval<F>& rhs)
{
  return lhs.inf() == (T) rhs.inf() && lhs.sup() == (T) rhs.sup();
}


template <class T, class F> inline __device__ __host__
bool operator==(const Interval<T>& lhs, const CudaInterval<F>& rhs)
{
  return lhs.inf() == (T) rhs.inf() && lhs.sup() == (T) rhs.sup();
}


template <class T> inline __device__ __host__
bool operator==(const CudaInterval<T>& lhs, const double& rhs)
{
  return lhs.inf() == (T) rhs && lhs.sup() == (T) rhs;
}

template <class T> inline __device__ __host__
bool operator==(const double& lhs, const CudaInterval<T>& rhs)
{
  return rhs.inf() == (T) lhs && rhs.sup() == (T) lhs;
}

template <class T> inline __device__ __host__
bool operator==(const CudaInterval<T>& lhs, const float& rhs)
{
  return lhs.inf() == (T) rhs && lhs.sup() == (T) rhs;
}

template <class T> inline __device__ __host__
bool operator==(const float& lhs, const CudaInterval<T>& rhs)
{
  return rhs.inf() == (T) lhs && rhs.sup() == (T) lhs;
}

template <class T, class F> inline __device__ __host__
bool operator!=(const CudaInterval<T>& lhs, const F& rhs)
{
  return !(lhs == rhs);
}


// Assignment
template <class T>
template <class F> inline __device__ __host__
CudaInterval<T>& CudaInterval<T>::operator=(const CudaInterval<F> &rhs)
{
    inf_ = (T) rhs.inf();
    sup_ = (T) rhs.sup();
    return *this;
}


template <class T>
template <class F> inline __host__
CudaInterval<T>& CudaInterval<T>::operator=(const Interval<F> &rhs)
{
    inf_ = (T) rhs.inf();
    sup_ = (T) rhs.sup();
    return *this;
}


template <class T> inline __device__ __host__
CudaInterval<T>& CudaInterval<T>::operator=(const double &rhs)
{
    inf_ = sup_ = (T) rhs;
    return *this;
}


} // namespace ilag

#endif



