//------------------------------------------------------------------------------
// Copyright (c) 2013 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

//This software contains source code provided by NVIDIA Corporation.



#ifndef IntervalCR_LIB_H
#define IntervalCR_LIB_H

#include <stdio.h>
#include "IntervalCR.h"


namespace intlag {



// Arithmetic operations

// Unary operators
template<class T> inline __device__
IntervalCR<T> const &operator+(IntervalCR<T> const &x)
{
    return x;
}

template<class T> inline __device__
IntervalCR<T> operator-(IntervalCR<T> const &x)
{
    return IntervalCR<T>(-x.center(), x.radius());
}

// Binary operators



						
}

} // namespace ilag
#endif



