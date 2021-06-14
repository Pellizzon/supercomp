#include <iostream>
#include <omp.h>

int main()
{
#pragma omp parallel
    {
#pragma omp master
        {
            std::cout << "só roda uma vez na thread:" << omp_get_thread_num() << "\n";
#pragma omp task
            {
                std::cout << "Estou rodando na thread:" << omp_get_thread_num() << "\n";
            }
        }
    }

    return 0;
}