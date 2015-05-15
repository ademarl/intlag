//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------

#ifndef ROUND_STATUS_H
#define ROUND_STATUS_H

#include<fenv.h>
#include<algorithm>
#include<cmath>

#define RS_UNKNOWN (abs(FE_UPWARD) + abs(FE_TOWARDZERO) + abs(FE_TOWARDZERO) + abs(FE_TOWARDZERO))

// A wrapper for rounding mode intrinsics, implemented with fenv.h 

namespace intlag {


namespace RoundStatus {
    const short down    = FE_DOWNWARD;
    const short near    = FE_TONEAREST;
    const short up      = FE_UPWARD;
    const short chop    = FE_TOWARDZERO;
    const short unknown  = RS_UNKNOWN;
}


inline static short getRoundMode() {
  switch(fegetround())
  {
    case FE_UPWARD: return RoundStatus::up;
    case FE_DOWNWARD: return RoundStatus::down;
    case FE_TONEAREST: return RoundStatus::near;
    case FE_TOWARDZERO: return RoundStatus::chop;
    default: return RoundStatus::unknown;
  }
}

inline static short setRoundMode(short rs) {
    return fesetround(rs);
}


} // namespace intlag

#endif
