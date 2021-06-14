#include <iostream>
#include <random>

// g++ -O3 -Wall pi_montecarlo_par_seed.cpp -o paralelo2 -fopenmp && time ./paralelo2

int main()
{
    double sum = 0.0;
    int N = 100000;
    
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++)
    {
        std::default_random_engine generator(i);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        double x = distribution(generator);
        double y = distribution(generator);

        if (x * x + y * y <= 1)
            sum += 1;
    }

    double pi = 4 * sum / N;

    // std::cerr << sum << "\n";

    std::cout << pi  << "\n";

    return pi;
}