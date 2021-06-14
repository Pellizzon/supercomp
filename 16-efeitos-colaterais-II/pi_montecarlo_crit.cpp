#include <iostream>
#include <random>

// g++ -O3 -Wall pi_montecarlo_crit.cpp -o paralelo -fopenmp && time ./paralelo

int main()
{
    double sum = 0.0;
    int N = 100000;
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++)
    {
        double x, y;
        #pragma omp critical
        {
            x = distribution(generator);
            y = distribution(generator);
        }

        if (x * x + y * y <= 1)
            sum += 1;
    }

    double pi = 4 * sum / N;

    // std::cerr << sum << "\n";
    
    std::cout << pi << "\n";

    return pi;
}