#include <omp.h>
#include <iostream>

// g++ -O3 exemplo3.cpp -o exemplo3 -fopenmp && ./exemplo3

int main()
{
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < 16; i++)
    {
        std::cout << i << " Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    return 0;
}
