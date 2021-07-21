#include <iostream>
#include <chrono>
#include <unistd.h>

// g++ -O3 q2-paralelo.cpp -o q2 -fopenmp && ./q2

void func1()
{
    sleep(2);
}

void func2()
{
    sleep(7);
}

void func3()
{
    sleep(3);
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp master
        {
            #pragma omp task
            {
                func1();
            }   
            #pragma omp task
            {
                func2();
            }
            #pragma omp task
            {
                func3();
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << duration.count() << "s\n";

    return 0;
}