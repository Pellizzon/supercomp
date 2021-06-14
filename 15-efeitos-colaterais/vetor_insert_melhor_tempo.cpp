#include <vector>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

// g++ -O3 vetor_insert_melhor_tempo.cpp -o t2 -fopenmp && time ./t2

double conta_complexa(int i)
{
	sleep(1);
	return 2 * i;
}

int main()
{
	int N = 10;
	std::vector<double> vec;
	#pragma omp parallel for default(none) shared(vec) shared(N)
	for (int i = 0; i < N; i++)
	{
		double valor = conta_complexa(i);
		#pragma omp critical
		{
			// código aqui dentro roda somente em uma thread por vez.
			vec.push_back(valor);
		}
	}

	for (int i = 0; i < N; i++)
	{
		std::cout << i << " ";
	}

	return 0;
}
