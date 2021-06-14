#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>

// nvcc -O3 -std=c++14 example3.cu -o t3 && ./t3 < stocks.txt

struct is_positive
{
    __host__ __device__ bool operator()(const double &x)
    {
        return x > 0;
    }
};

struct is_negative
{
    __host__ __device__ bool operator()(const double &x)
    {
        return x < 0;
    }
};

int main()
{
    thrust::host_vector<double> host;

    int N = 0;

    while (std::cin.fail() == false)
    {
        N += 1;

        double val;

        std::cin >> val;
        host.push_back(val);
    }

    /* na linha abaixo os dados são copiados para GPU */
    thrust::device_vector<double> stocks(host);
    thrust::device_vector<double> ganho_diario(N - 1);

    thrust::transform(stocks.begin() + 1, stocks.end(), stocks.begin(), ganho_diario.begin(), thrust::minus<double>());

    // for (auto i = stocks.begin(); i != stocks.end(); i++)
    // {
    //     std::cout << *i << " "; // este acesso é rápido -- CPU
    // }
    // printf("\n");

    // for (auto i = ganho_diario.begin(); i != ganho_diario.end(); i++)
    // {
    //     std::cout << *i << " "; // este acesso é rápido -- CPU
    // }
    // printf("\n");

    int Npositive = thrust::count_if(ganho_diario.begin(), ganho_diario.end(), is_positive());

    thrust::replace_if(ganho_diario.begin(), ganho_diario.end(), ganho_diario.begin(), is_negative(), 0);

    // for (auto i = ganho_diario.begin(); i != ganho_diario.end(); i++)
    // {
    //     std::cout << *i << " "; // este acesso é rápido -- CPU
    // }
    // printf("\n");

    double positivesSum = thrust::reduce(ganho_diario.begin(), ganho_diario.end(), 0.0, thrust::plus<double>());

    double positivesMean = positivesSum / Npositive;

    std::cout << "quantas vezes o valor subiu? " << Npositive << "\n";
    std::cout << "qual é o aumento médio, considerando só as vezes em que o valor aumentou de fato? " << positivesMean << "\n";
}
