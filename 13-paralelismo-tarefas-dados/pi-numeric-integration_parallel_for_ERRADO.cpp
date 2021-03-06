#include <iostream>
#include <chrono>

static long num_steps = 1000000000;
double step;

// g++ -O3 pi-numeric-integration_parallel_for_ERRADO.cpp -o t4 -fopenmp && time ./t4
// 1,474 sem paralelizar
// paralelizando
int main()
{
    int i;
    double pi, sum = 0.0;
    step = 1.0 / (double)num_steps;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (i = 0; i < num_steps; i++)
    {
        double x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " segundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
}