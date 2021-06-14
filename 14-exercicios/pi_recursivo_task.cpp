#include <omp.h>
#include <iostream>
#include <iomanip>
static long num_steps = 1024l * 1024 * 1024 * 2;

#define MIN_BLK 1024 * 1024 * 256

// g++ -O3 pi_recursivo_task.cpp -o t3 -fopenmp && ./t3

double pi_r(long Nstart, long Nfinish, double step)
{
    double sum = 0;

    if (Nfinish - Nstart < MIN_BLK)
    {
        for (long i = Nstart; i < Nfinish; i++)
        {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
    }
    else
    {
        #pragma omp parallel
        {
            #pragma omp master
            {
                long iblk = Nfinish - Nstart;
                double sum1, sum2;

                #pragma omp task shared(sum1)
                {
                    sum1 = pi_r(Nstart, Nfinish - iblk / 2, step);
                }

                #pragma omp task shared(sum2)
                {
                    sum2 = pi_r(Nfinish - iblk / 2, Nfinish, step);
                }

                #pragma omp taskwait
                    sum = sum1 + sum2;
            }
        }
    }

    return sum;
}

int main()
{
    long i;
    double step, pi;
    double init_time, final_time;
    step = 1.0 / (double)num_steps;
    init_time = omp_get_wtime();
    double sum;

    sum = pi_r(0, num_steps, step);

    pi = step * sum;
    final_time = omp_get_wtime() - init_time;

    std::cout << "for " << num_steps << " steps pi = " << std::setprecision(15) << pi << " in " << final_time << " secs\n";
}
