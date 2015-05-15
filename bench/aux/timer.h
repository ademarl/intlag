
#ifndef TIMER_AUX_H
#define TIMER_AUX_H


#include <sys/time.h>

class Timer {

	public:

		Timer() {
			gettimeofday(&start_tv, NULL);
		}

		void reset() {
			gettimeofday(&start_tv, NULL);
		}

		void stop() {
			gettimeofday(&tv, NULL);
		}

		double getSeconds() {
			return (tv.tv_sec - start_tv.tv_sec) + (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
		}


	private:

	struct timeval tv;
	struct timeval start_tv;

};

#endif
