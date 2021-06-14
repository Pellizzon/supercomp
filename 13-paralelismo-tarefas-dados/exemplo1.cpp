#include <omp.h>
#include <iostream>

// g++ -O3 exemplo1.cpp -o exemplo1 -fopenmp && ./exemplo1

int main()
{
#pragma omp parallel for
    for (int i = 0; i < 16; i++)
    {
        std::cout << i << " Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    return 0;
}