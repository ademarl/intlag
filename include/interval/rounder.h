


// Wrapper for CPU rounded arithmetic which uses <method>


#ifndef ROUNDER_H
#define ROUNDER_H


#include <algorithm>
#include <cmath>
#include <gmp.h>
#include <mpfr.h>


#include "round_status.h"


namespace intlag {


template<class F>
class Rounder {

  typedef int mpfr_func(mpfr_t, const __mpfr_struct*, mp_rnd_t);
  inline static float invoke_mpfr(float x, mpfr_func f, mp_rnd_t r)
  {
    //float x = y;
    mpfr_t xx;
    mpfr_init_set_d(xx, x, GMP_RNDN); // use r as mode for this?
    f(xx, xx, r);
    float res = mpfr_get_flt(xx, r);
    mpfr_clear(xx);
    return res;
  }
  inline static double invoke_mpfr(double x, mpfr_func f, mp_rnd_t r)
  {
    //double x = y;
    mpfr_t xx;
    mpfr_init_set_d(xx, x, GMP_RNDN); // use r as mode for this?
    f(xx, xx, r);
    double res = mpfr_get_d(xx, r);
    mpfr_clear(xx);
    return res;
   }

  public:

    // chop and near

    //static F supremum();
    //static F infimun();

    //static F ulp_up(const F x);
    //static F ulp_down(const F x);

  inline static F nan()
  {
    return (F) nanf("");
  }


  inline static F add_up(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::up);
    F result = x + y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F add_down(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::down);
    F result = x + y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F sub_up(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::up);
    F result = x - y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F sub_down(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::down);
    F result = x - y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F mul_up(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::up);
    F result = x * y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F mul_down(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::down);
    F result = x * y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F div_up(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::up);
    F result = x / y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F div_down(const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::down);
    F result = x / y;
    setRoundMode(oldstatus);
    return result;
  }

  inline static F fma_up(const F a, const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::up);
    F result = fma(a, x, y);
    setRoundMode(oldstatus);
    return result;
  }

  inline static F fma_down(const F a, const F x, const F y)
  {
    short oldstatus = getRoundMode();
    setRoundMode(RoundStatus::down);
    F result = fma(a, x, y);
    setRoundMode(oldstatus);
    return result;
  }

  inline static F sqrt_up(const F x)
  {
    return invoke_mpfr(x, mpfr_sqrt, GMP_RNDU);
  }

  inline static F sqrt_down(const F x)
  {
    return invoke_mpfr(x, mpfr_sqrt, GMP_RNDD);
  }

  inline static F int_up(const F x)
  {
    return ceil(x);
  }

  inline static F int_down(const F x)
  {
    return floor(x);
  }
};

} // namespace intlag

#endif



