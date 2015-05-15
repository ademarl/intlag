//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

//This software contains source code provided by NVIDIA Corporation.


#ifndef IntervalCR_H
#define IntervalCR_H

#include <cstdio>
#include "cuda_rounder.h"

// Interval template class and basic operations
// Interface inspired from the Boost Interval library (www.boost.org)

namespace intlag {

template <class T>
class IntervalCR
{
    public:
        __device__ __host__ IntervalCR();
        __device__ __host__ IntervalCR(T const &v);
        __device__ __host__ IntervalCR(T const &l, T const &u);

        __device__ __host__ T const &center() const;
        __device__ __host__ T const &radius() const;

        static __device__ __host__ IntervalCR empty();

    private:
        T c;
        T r;
};

// Constructors
template<class T> inline __device__ __host__
IntervalCR<T>::IntervalCR()
{
}

template<class T> inline __device__ __host__
IntervalCR<T>::IntervalCR(T const &center, T const &radius) :
    c(center), r(radius)
{
}

template<class T> inline __device__ __host__
IntervalCR<T>::IntervalCR(T const &center) :
    c(center), r(0.0)
{
}


template<class T> inline __device__ __host__
T const &IntervalCR<T>::center() const
{
    return c;
}

template<class T> inline __device__ __host__
T const &IntervalCR<T>::radius() const
{
    return r;
}

template<class T> inline __device__ __host__
IntervalCR<T> IntervalCR<T>::empty()
{
    rounded_arith<T> rnd;
    return IntervalCR<T>(rnd.nan(), rnd.nan());
}


template<class T> inline __device__
IntervalCR<T> operator+(IntervalCR<T> const x, IntervalCR<T> const y)
{
    rounded_arith<T> rnd;
		T center = rnd.add_rz(x.center(), y.center());
    return IntervalCR<T>( center,
                            rnd.add_up( rnd.add_up(x.radius(), y.radius()), ulp_up(fabs(center)) ) );
}

// Multiplication
template<class T> inline __device__
IntervalCR<T> operator*(IntervalCR<T> const x, IntervalCR<T> const y)
{
    rounded_arith<T> rnd;
		T center = rnd.mul_rz(x.center(), y.center());
    return IntervalCR<T>( center,
														rnd.add_up(
															rnd.add_up(
																rnd.add_up(
																	rnd.mul_up(x.radius(), y.radius() ),
																	rnd.mul_up(x.radius(), fabs(y.center()))),
																rnd.mul_up(y.radius(), fabs(x.center()))),
															scalbln(fabs(center), -52) )  ); // folga pela aproximação do centro
}


/* SUM
template<class T> inline __device__
IntervalCR<T> operator+(IntervalCR<T> const &x, IntervalCR<T> const &y)
{
    rounded_arith<T> rnd;
		T center = rnd.add_rz(x.center(), y.center());
    return IntervalCR<T>( center,
                            rnd.add_up( rnd.add_up(x.radius(), y.radius()), scalbln(fabs(center), -52) ) );
}

// Multiplication
template<class T> inline __device__
IntervalCR<T> operator*(IntervalCR<T> const &x, IntervalCR<T> const &y)
{
    rounded_arith<T> rnd;
		T center = rnd.mul_rz(x.center(), y.center());
    return IntervalCR<T>( center,
														rnd.add_up(
															rnd.add_up(
																rnd.add_up(
																	rnd.mul_up(x.radius(), y.radius() ),
																	rnd.mul_up(x.radius(), fabs(y.center()))),
																rnd.mul_up(y.radius(), fabs(x.center()))),
															scalbln(fabs(center), -52) )  ); // folga pela aproximação do centro
*/

} // namespace ilag





#endif
