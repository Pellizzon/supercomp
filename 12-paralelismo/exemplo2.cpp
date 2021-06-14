#include <iostream>
#include <omp.h>

int main()
{
    int res = 1;
#pragma omp parallel
    {
        for (int i = 0; i < 10000; i++)
        {
            res += 1;
        }
    }
    std::cout << res << "\n";
    return 0;
}