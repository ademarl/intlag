
//------------------------------------------------------------------------------
// Copyright (c) 2014 Ademar Marques Lacerda Filho
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//------------------------------------------------------------------------------


#ifndef CASE_H
#define CASE_H

#include <map>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>

#include "timer.h"
#include "gtest/gtest.h"

// Defines the macros BENCH and BENCH_F for benchmark testing, and their
//auxiliar methods
// The behaviour of these macros is to create tests from 'googletest' which time
// an specific action and keeps the result on a file called 'benchmark.txt'


extern std::map<std::string, double> timings;


namespace intlag
{
namespace bench
{




// Base class for Benchmark Cases
class BenchTest : public ::testing::Test {
  protected:

    void SetUp() {}
    void TearDown() {}
    void TestBody() {}

  public:
    // Test Case methods
    void begin() {}
    void run() {}
    void check() {}
    void end() {}
    short iterations() {return 100;}
};


// Benchmark Test (used by BENCH and BENCH_F macros)
template <class Case>
class BenchCase {
  private:
    Case test_case;
    char *s, *n, *type;

  public:

    BenchCase(char* suite, char* name, char* type) :s(suite), n(name), type(type) {}

    void run() {
      // setup
      test_case.SetUp();
      test_case.begin();


      // timed run
      short N = test_case.iterations();
      Timer t;
      for(short k = 0; k < N; ++k)
        test_case.run();
      t.stop();


      // store in global map structure minding the number of repetitions 
      char buffer[30]; sprintf(buffer, "(%d times)", N);
      std::string key(s); key += '.'; key += n; key += '.'; key += type;
      key += buffer;
      if (timings.find(key) == timings.end())
        timings.insert(std::pair<std::string, double>(key, t.getSeconds()));
      else {
        (timings.find(key))->second += t.getSeconds();
      }


      // tear down
      test_case.check();
      test_case.end();
      test_case.TearDown();
    }
};


//TODO: Compartimentalize this
#define BENCH_F_F(Suite, Name, Case)                                \
TEST_F(Suite, Name ## Float)                                               \
{                                                                 \
  BenchCase< Case<float> > bench_case_f(#Suite, #Name , "Float");    \
  bench_case_f.run();                                             \
}   
//------------------------------------------------------------------------------
#define BENCH_D_F(Suite, Name, Case)                                \
TEST_F(Suite, Name ## Double)                                               \
{                                                                   \
  BenchCase< Case<double> > bench_case_d(#Suite, #Name, "Double");  \
  bench_case_d.run();                                             \
} 
//------------------------------------------------------------------------------
#define BENCH_FD_F(Suite, Name, Case)                                \
TEST_F(Suite, Name ## Float)                                               \
{                                                                 \
  BenchCase< Case<float> > bench_case_f(#Suite, #Name , "Float");    \
  bench_case_f.run();                                             \
}                                                                 \
TEST_F(Suite, Name ## Double)                                               \
{                                                                   \
  BenchCase< Case<double> > bench_case_d(#Suite, #Name, "Double");  \
  bench_case_d.run();                                             \
}
//------------------------------------------------------------------------------
#define BENCH_FD(Suite, Name, Case)                                  \
TEST(Suite, Name ## Float)                                                 \
{                                                                 \
  BenchCase< Case<float> > bench_case_f(#Suite, #Name, "Float");    \
  bench_case_f.run();                                             \
}                                                                 \
TEST(Suite, Name ## Double)                                                 \
{                                                                 \
  BenchCase< Case<double> > bench_case_d(#Suite, #Name, "Double");  \
  bench_case_d.run();                                             \
}

} // namespace bench
} // namespace intlag

#endif
