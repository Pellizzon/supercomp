#include <vector>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

// g++ -O3 vetor_insert_sem_pushback.cpp -o t3 -fopenmp && time ./t3

double conta_complexa(int i)
{
	sleep(1);
	return 2 * i;
}

int main()
{
	int N = 10;
	std::vector<double> vec(N);
	#pragma omp parallel for default(none) shared(vec) shared(N)
	for (int i = 0; i < N; i++)
	{
		vec[i] = conta_complexa(i);
	}

	for (int i = 0; i < N; i++)
	{
		std::cout << i << " ";
	}

	return 0;
}
