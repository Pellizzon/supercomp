#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>

// nvcc -O3 -std=c++14 tarefa1.cu -o t1 && ./t1 < stocks2.csv

struct variance
{
    double mean;
    int N;

    variance(double a, int n) : mean(a), N(n){};
    __host__ __device__ double operator()(const double &x)
    {
        return ((x - mean) * (x - mean)) / N;
    }
};

int main()
{
    thrust::host_vector<double> hostApple;
    thrust::host_vector<double> hostMicrosoft;

    int N = 0;

    while (std::cin.fail() == false)
    {
        N += 1;

        double aapl, msft;

        std::cin >> aapl >> msft;
        hostApple.push_back(aapl);
        hostMicrosoft.push_back(msft);
    }

    /* na linha abaixo os dados são copiados para GPU */
    thrust::device_vector<double> AAPL(hostApple);
    thrust::device_vector<double> MSFT(hostMicrosoft);
    thrust::device_vector<double> diff(N);

    thrust::transform(MSFT.begin(), MSFT.end(), AAPL.begin(), diff.begin(), thrust::minus<double>());

    double mean = thrust::reduce(diff.begin(), diff.end(), 0.0, thrust::plus<double>()) / N;
    std::cout << "Média da diferença das ações: " << mean << "\n";

    thrust::device_vector<double> xMinusMeanSquared(N);
    thrust::transform(diff.begin(), diff.end(), xMinusMeanSquared.begin(), variance(mean, N));

    double variance = thrust::reduce(xMinusMeanSquared.begin(), xMinusMeanSquared.end(), 0.0, thrust::plus<double>());
    std::cout << "Variância: " << variance << "\n";
}
