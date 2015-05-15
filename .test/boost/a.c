
#include <iostream>

#include <boost/numeric/interval.hpp>
using boost::numeric::interval;
using namespace boost::numeric;

int main() {

	interval<double> x(-2, 70), y(-55, -50);
	interval<double> z = x*y;


	std::cout << z.lower() << ", " << z.upper() << std::endl;

	return 0;
}
