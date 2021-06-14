#include <iostream>
#include <chrono>

static long num_steps = 1000000000;
double step;

// g++ -O3 pi-numeric-integration.cpp -o t1 -fopenmp && time ./t1

int main()
{
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
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
