
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef INTERVAL_LIB_H
#define INTERVAL_LIB_H

#include <math.h>
#include "interval.h"
#include "../aux/min_max.h"

namespace intlag {


template <class T> inline
bool contain_zero(Interval<T> const x)
{
  if (x.inf() <= 0 && x.sup() >= 0) return true;
  return false;
}


template <class T> inline
bool empty(Interval<T> const x)
{
  if (isnan(x.inf()) || isnan(x.sup())) return true;
  return false;
}


template <class T> inline
Interval<T> abs(Interval<T> const x)
{
  if (empty(x))
    return Rounder<T>::nan();
  return Interval<T>( max(x.inf(), -x.sup())*(!contain_zero(x)), max(-x.inf(), x.sup()) );
}


template <class T> inline
T width(Interval<T> x)
{
  if (empty(x))
    return 0;
  return Rounder<T>::sub_up(x.sup(), x.inf());
}


// Median - rounded by current round mode
template <class T> inline
T median(Interval<T> const x)
{
  if (empty(x))
    return Rounder<T>::nan();
  return Rounder<T>::div_up(width(x), 2.0) + x.inf();
}


// Median - rounded up
template <class T> inline
T median_up(Interval<T> const x)
{
  if (empty(x))
    return Rounder<T>::nan();
  return Rounder<T>::add_up(Rounder<T>::div_up(width(x), 2.0), x.inf());
}


// Median - rounded down
template <class T> inline
T median_down(Interval<T> const x)
{
  if (empty(x))
    return Rounder<T>::nan();
  return Rounder<T>::add_down(Rounder<T>::div_up(width(x), 2.0), x.inf());
}


// Intersection
template <class T> inline
Interval<T> overlap(Interval<T> const x, Interval<T> const y)
{
  if(empty(x) || empty(y)) return Interval<T>::empty();

  T u = max(x.inf(), y.inf());
  T v = min(x.sup(), y.sup());

  if (u <= v)
    return Interval<T>(u, v);
  return Interval<T>::empty();
}


// Hull
template <class T> inline
Interval<T> hull(Interval<T> const x, Interval<T> const y)
{
  if(empty(x) || empty(y)) return Interval<T>::empty();

  T u = min(x.inf(), y.inf());
  T v = max(x.sup(), y.sup());

  if (u <= v)
    return Interval<T>(u, v);
  return Interval<T>::empty();
}


// Unary
template <class T> inline
Interval<T> operator+(Interval<T> const x)
{
    return x;
}


template <class T> inline
Interval<T> operator-(Interval<T> const x)
{
    return Interval<T>(-x.sup(), -x.inf());
}


// Binary
//TODO: Operations with numbers
template <class T> inline
Interval<T> operator+(Interval<T> const x, Interval<T> const y)
{
    return Interval<T>( Rounder<T>::add_down(x.inf(), y.inf()),
                        Rounder<T>::add_up(x.sup(), y.sup())   );
}


template <class T, class F> inline
Interval<T> operator+(Interval<T> const x, F const y)
{
    return Interval<T>( Rounder<T>::add_down(x.inf(), y),
                        Rounder<T>::add_up(x.sup(), y)   );
}


template <class T, class F> inline
Interval<T> operator+(F const x, Interval<T> const y)
{
    return Interval<T>( Rounder<T>::add_down(x, y.inf()),
                        Rounder<T>::add_up(x, y.sup())   );
}


template <class T> inline
Interval<T> operator-(Interval<T> const x, Interval<T> const y)
{
    return Interval<T>( Rounder<T>::sub_down(x.inf(), y.sup()),
                        Rounder<T>::sub_up(x.sup(), y.inf())   );
}

template <class T, class F> inline
Interval<T> operator-(Interval<T> const x, F const y)
{
    return Interval<T>( Rounder<T>::sub_down(x.inf(), y),
                        Rounder<T>::sub_up(x.sup(), y)   );
}


template <class T, class F> inline
Interval<T> operator-(F const x, Interval<T> const y)
{
    return Interval<T>( Rounder<T>::sub_down(x, y.sup()),
                        Rounder<T>::sub_up(x, y.inf())   );
}


template <class T, class F> inline
Interval<T> operator*(F const a, Interval<T> const x)
{
    return Interval<T>(  min( Rounder<T>::mul_down(a, x.inf()),
                              Rounder<T>::mul_down(a, x.sup()) ),
                         max( Rounder<T>::mul_up(a, x.inf()),
                              Rounder<T>::mul_up(a, x.sup())   ));
}


template <class T, class F> inline
Interval<T> operator*(Interval<T> const x, F const a)
{
    return Interval<T>(  min( Rounder<T>::mul_down(a, x.inf()),
                              Rounder<T>::mul_down(a, x.sup()) ),
                         max( Rounder<T>::mul_up(a, x.inf()),
                              Rounder<T>::mul_up(a, x.sup())   ));
}


template <class T> inline
Interval<T> operator*(Interval<T> const x, Interval<T> const y)
{
  return Interval<T>(  min( Rounder<T>::mul_down(x.inf(), y.inf()),
                            Rounder<T>::mul_down(x.inf(), y.sup()),
                            Rounder<T>::mul_down(x.sup(), y.inf()),
                            Rounder<T>::mul_down(x.sup(), y.sup()) ),
                       max( Rounder<T>::mul_up(x.inf(), y.inf()),
                            Rounder<T>::mul_up(x.inf(), y.sup()),
                            Rounder<T>::mul_up(x.sup(), y.inf()),
                            Rounder<T>::mul_up(x.sup(), y.sup())) );
}


template <class T> inline
Interval<T> operator/(Interval<T> const x, Interval<T> const y)
{
  if(contain_zero(y)) return Interval<T>::empty();

  return Interval<T>(  min( Rounder<T>::div_down(x.inf(), y.inf()),
                            Rounder<T>::div_down(x.inf(), y.sup()),
                            Rounder<T>::div_down(x.sup(), y.inf()),
                            Rounder<T>::div_down(x.sup(), y.sup()) ),
                       max( Rounder<T>::div_up(x.inf(), y.inf()),
                            Rounder<T>::div_up(x.inf(), y.sup()),
                            Rounder<T>::div_up(x.sup(), y.inf()),
                            Rounder<T>::div_up(x.sup(), y.sup())) );
}


template <class T, class F> inline
Interval<T> operator/(Interval<T> const x, F const y)
{
  if(y == 0.0) return Interval<T>::empty();

  return Interval<T>(  min( Rounder<T>::div_down(x.inf(), y),
                            Rounder<T>::div_down(x.sup(), y) ),
                       max( Rounder<T>::div_up(x.inf(), y),
                            Rounder<T>::div_up(x.sup(), y)) );
}


template <class T, class F> inline
Interval<T> operator/(F const x, Interval<T> const y)
{
  if(contain_zero(y)) return Interval<T>::empty();

  return Interval<T>(  min( Rounder<T>::div_down(x, y.inf()),
                            Rounder<T>::div_down(x, y.sup()) ),
                       max( Rounder<T>::div_up(x, y.inf()),
                            Rounder<T>::div_up(x, y.sup())) );
}


// Other methods
template <class T, class F> inline
Interval<T> fma(F const a, Interval<T> const x, Interval<T> const y)
{
    return Interval<T>(  min( Rounder<T>::fma_down(a, x.inf(), y.inf()),
                              Rounder<T>::fma_down(a, x.sup(), y.inf()) ),
                         max( Rounder<T>::fma_up(a, x.inf(), y.sup()),
                              Rounder<T>::fma_up(a, x.sup(), y.sup())   ));
}


template <class T> inline
Interval<T> sqrt(Interval<T> const x)
{
	return Interval<T>(	Rounder<T>::sqrt_down( x.inf() ), Rounder<T>::sqrt_up( x.sup() ) );
}


} // namespace intlag

#endif



