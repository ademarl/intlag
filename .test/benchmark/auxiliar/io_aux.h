


#ifndef IO_AUX_H
#define IO_AUX_H

#include <iostream>
using namespace std;

// Sets method and number of iterations from command line arguments
void set_options(int argc, char** argv, int* method, int* iterations){

	switch(argc){
		case 1:
			break;

		case 3:
			*iterations = atoi(argv[2]);

		case 2:
			*method = atoi(argv[1]);
			break;

		default:
			*method = atoi(argv[1]);
			*iterations = atoi(argv[2]);
	}
}

// Reads N intervals for x, and another N for y from the stdin
template <class T>
void read_interval_input(T* x, T* y, int N) {

	double aux, aux2;
	for (int i = 0; i < N; ++i){
		cin >> aux; cin >> aux2;
		x[i] = T(aux, aux2);
	}
	for (int i = 0; i < N; ++i){
		cin >> aux; cin >> aux2;
		y[i] = T(aux, aux2);
	}
}


#endif
