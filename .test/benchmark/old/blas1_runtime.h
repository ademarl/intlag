
#ifndef BLAS1_RUNTIME_H
#define BLAS1_RUNTIME_H

#include <cstdio>
#include <cstdlib>
using namespace std;

#include<omp.h>
#include <boost/numeric/interval.hpp>
using boost::numeric::interval;
using namespace boost::numeric;

#include "box_blas.h"
//#include "cuda_box_lib.h"
//#include "cuda_interval_lib.h"

#include "test_cases.h"
#include "test_cases_serial.h"
#include "test_cases_omp.h"


#include "io_aux.h"
#include "timer_aux.h"

// NOTE: Test cases ignore the return value of expressions by not assigning the results

// TODO: Refactor input reading
// TODO: Refactor test cases of cuda cases into a function that receives a fpointer (needs a wrapper for box_gpu_lib.h)



// Serial runtime tests
void serial_blas1_runtime(int n_iter){

	cout << "Using serial implementation..." << endl;

	int n;
	double alpha;
	interval<double> *x = NULL, *y = NULL;

	// Reads input
	cin >> alpha;
	cin >> n;
	x = new interval<double>[n];
	y = new interval<double>[n];
	read_interval_input(x, y, n);

	Timer t;

	// Test cases
	t = test_case(n_iter, n, alpha, x, &serial_scal);
	cout << "Time for the execution of " << n << " interval scalar multiplications " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &serial_sum);
	cout << "Time for the execution of " << n << " interval sums " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &serial_mult);
	cout << "Time for the execution of " << n << " interval multiplications " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, alpha, x, y, &serial_axpy);
	cout << "Time for the execution of " << n << " interval axpy's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, &serial_asum);
	cout << "Time for the execution of " << n << " interval asum's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &serial_dot);
	cout << "Time for the execution of " << n << " interval dot products " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &serial_norm2);
	cout << "Time for the execution of " << n << " interval norm2 " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	delete[] x;
	delete[] y;
}

// OpenMP runtime tests
void openmp_blas1_runtime(int n_iter) {

	omp_set_num_threads(8);
	cout << "Using OpenMP implementation..." << endl;

	int n;
	double alpha;
	interval<double> *x = NULL, *y = NULL;

	// Reads input
	cin >> alpha;
	cin >> n;
	x = new interval<double>[n];
	y = new interval<double>[n];
	read_interval_input(x, y, n);

	Timer t;

	// Test cases
	t = test_case(n_iter, n, alpha, x, &omp_scal);
	cout << "Time for the execution of " << n << " interval scalar multiplications " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &omp_sum);
	cout << "Time for the execution of " << n << " interval sums " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &omp_mult);
	cout << "Time for the execution of " << n << " interval multiplications " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, alpha, x, y, &omp_axpy);
	cout << "Time for the execution of " << n << " interval axpy's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, &omp_asum);
	cout << "Time for the execution of " << n << " interval asum's " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &omp_dot);
	cout << "Time for the execution of " << n << " interval dot products " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	t = test_case(n_iter, n, x, y, &omp_norm2);
	cout << "Time for the execution of " << n << " interval norm2 " << n_iter << " times: " << t.getSeconds() << "s" << endl;

	delete[] x;
	delete[] y;

}

// CUDA runtime tests
void cuda_blas1_runtime(int n_iter) {
	cout << "Using CUDA implementation..." << endl;

	int n;
	double alpha;

	// Reads input
	cin >> alpha;
	cin >> n;
	box_gpu<double> box_x(n), box_y(n);
	read_interval_input(box_x.data(), box_y.data(), n);


	// Running 'n_iter' times the sum of boost intervals: x*a
	StopWatchInterface* t = start_timer();
	for (int i = 0; i < n_iter; ++i)
		box_x*alpha;
	end_timer(t);
	cout << "Time for the execution of " << n << " interval scalar multiplications " << n_iter << " times: " << sdkGetTimerValue(&t)/1000 << "s" << endl;

	// Running 'n_iter' times the sum of boost intervals: x+y
	t = start_timer();
	for (int i = 0; i < n_iter; ++i)
		box_x + box_y;
	end_timer(t);
	cout << "Time for the execution of " << n << " interval sums " << n_iter << " times: " << sdkGetTimerValue(&t)/1000 << "s" << endl;


}

#endif
