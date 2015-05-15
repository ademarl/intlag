
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef INTERVAL_H
#define INTERVAL_H

#include <assert.h>
#include "rounder.h"

namespace intlag {


template <class T>
class Interval {

  public:
    Interval();
    Interval(T const x);
    Interval(T const x, T const y);

    T const &inf() const;
    T const &sup() const;

    static Interval empty();

    template <class F>
    Interval& operator=(const Interval<F> &rhs);
    template <class F>
    Interval& operator=(const F &rhs);

    Interval& operator+=(const Interval& rhs);
    Interval& operator-=(const Interval& rhs);
    Interval& operator*=(const Interval& rhs);
    Interval& operator/=(const Interval& rhs);

  private:
    T inf_;
    T sup_;
};



// Constructors
template <class T> inline 
Interval<T>::Interval() :
  inf_(Rounder<T>::nan()), sup_(Rounder<T>::nan())
{
}


template <class T> inline 
Interval<T>::Interval(T const x) :
  inf_(x), sup_(x)
{
}


template <class T> inline 
Interval<T>::Interval(T const x, T const y) :
  inf_(x), sup_(y)
{
  
  if (isnan(x) || isnan(y)) return;
  //if (x > y) printf("%.16f, %.16f\n", inf_, sup_);
  assert(inf_ <= sup_); // handle this error better
}


// Getters
template <class T> inline 
T const &Interval<T>::inf() const
{
    return inf_;
}


template <class T> inline 
T const &Interval<T>::sup() const
{
    return sup_;
}


template <class T> inline 
Interval<T> Interval<T>::empty()
{
    return Interval<T>(Rounder<T>::nan(), Rounder<T>::nan());
}


// Logical comparison
template <class T, class F> inline
bool operator==(const Interval<T>& lhs, const Interval<F>& rhs)
{
  return lhs.inf() == (T) rhs.inf() && lhs.sup() == (T) rhs.sup();
}


template <class T> inline
bool operator==(const Interval<T>& lhs, const double& rhs)
{
  return lhs.inf() == (T) rhs && lhs.sup() == (T) rhs;
}


template <class T> inline
bool operator==(const double& lhs, const Interval<T>& rhs)
{
  return rhs.inf() == (T) lhs && rhs.sup() == (T) lhs;
}


template <class T, class F> inline
bool operator!=(const Interval<T>& lhs, const F& rhs)
{
  return !(lhs == rhs);
}


template <class T> inline
bool operator!=(const double& lhs, const Interval<T>& rhs)
{
  return !(lhs == rhs);
}


// Assignments
template <class T>
template <class F> inline
Interval<T>& Interval<T>::operator=(const Interval<F> &rhs)
{
    inf_ = (T) rhs.inf();
    sup_ = (T) rhs.sup();
    return *this;
}


template <class T>
template <class F> inline
Interval<T>& Interval<T>::operator=(const F &rhs)
{
    inf_ = sup_ = (T) rhs;
    return *this;
}


template <class T> inline
Interval<T>& Interval<T>::operator+=(const Interval<T>& rhs){
  return *this = *this + rhs;
}


template <class T> inline
Interval<T>& Interval<T>::operator-=(const Interval<T>& rhs){
  return *this = *this - rhs;
}


template <class T> inline
Interval<T>& Interval<T>::operator*=(const Interval<T>& rhs){
  return *this = *this * rhs;
}


template <class T> inline
Interval<T>& Interval<T>::operator/=(const Interval<T>& rhs){
  return *this = *this / rhs;
}

} // namespace intlag

#endif



