#include <iostream>
#include <random>

// g++ -O3 -Wall pi_montecarlo.cpp -o sequencial && time ./sequencial

int main()
{
    double sum = 0.0;
    int N = 100000;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < N; i++)
    {
        double x = distribution(generator);
        double y = distribution(generator);

        if (x * x + y * y <= 1)
            sum += 1;
    }

    std::cout << 4 * sum / N << "\n";

    return 0;
}