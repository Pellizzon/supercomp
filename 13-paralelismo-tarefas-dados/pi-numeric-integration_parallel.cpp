#include <iostream>
#include <chrono>

static long num_steps = 1000000000;
double step;

// g++ -O3 pi-numeric-integration_parallel.cpp -o t2 -fopenmp && time ./t2
// 1,474 sem paralelizar
// paralelizando

int main()
{
    int i, j;
    double pi, sum1 = 0.0, sum2 = 0.0;
    step = 1.0 / (double)num_steps;

    double res_parte1, res_parte2;

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp master
        {
            #pragma omp task
            {
                for (i = 0; i < num_steps / 2; i++)
                {
                    double x1 = (i + 0.5) * step;
                    sum1 = sum1 + 4.0 / (1.0 + x1 * x1);
                }
            }
            #pragma omp task
            {
                for (j = num_steps / 2; j < num_steps; j++)
                {
                    double x2 = (j + 0.5) * step;
                    sum2 = sum2 + 4.0 / (1.0 + x2 * x2);
                }
            }
        }
    }

    pi = step * (sum1 + sum2);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " segundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
}
