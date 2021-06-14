#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>

// nvcc -O3 -std=c++14 example2.cu -o t2 && ./t2 < stocks2.csv

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

    thrust::device_vector<double> diff_X_minus_mean(N);
    thrust::transform(diff.begin(), diff.end(), thrust::constant_iterator<double>(mean), diff_X_minus_mean.begin(), thrust::minus<double>());

    thrust::device_vector<double> diff_X_minus_mean_squared(N);
    thrust::transform(diff_X_minus_mean.begin(), diff_X_minus_mean.end(), diff_X_minus_mean_squared.begin(), thrust::square<double>());
    double variance = thrust::reduce(diff_X_minus_mean_squared.begin(), diff_X_minus_mean_squared.end(), 0.0, thrust::plus<double>()) / N;
    std::cout << "Variância: " << variance << "\n";
}
