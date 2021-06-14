#include <vector>
#include <iostream>
#include <unistd.h>

// g++ -O3 vetor_insert.cpp -o t0 -fopenmp && time ./t0

double conta_complexa(int i)
{
	return 2 * i;
}

int main()
{
	int N = 10000;
	std::vector<double> vec;
	#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		// conflito
		vec.push_back(conta_complexa(i));
	}

	for (int i = 0; i < N; i++)
	{
		std::cout << i << " ";
	}

	return 0;
}
