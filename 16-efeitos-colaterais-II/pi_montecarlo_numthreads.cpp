#include <iostream>
#include <random>
#include <omp.h>
#include <vector>

// g++ -O3 -Wall pi_montecarlo_numthreads.cpp -o paralelo3 -fopenmp && time ./paralelo3

int main()
{
    double sum = 0.0;
    int N = 100000;
    int NUM_THREADS = omp_get_max_threads();

    std::vector<std::default_random_engine> generators(NUM_THREADS);
    for (int j = 0; j < NUM_THREADS; j++)
    {
        std::default_random_engine generator(j * 100);
        generators[j] = generator;
    }

    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++)
    {
        double x = distribution(generators[omp_get_thread_num()]);
        double y = distribution(generators[omp_get_thread_num()]);

        if (x * x + y * y <= 1)
            sum++;
    }

    double pi = 4 * sum / N;

    // std::cerr << sum << "\n";

    std::cout << pi << "\n";

    return pi;
}
