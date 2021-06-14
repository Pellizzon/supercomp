#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>
#include <chrono>

// nvcc -O3 -std=c++14 tarefa1.cu -o t1 && ./t1 < stocks.txt

int main()
{
    int N = 2518;

    thrust::host_vector<double> host(N);
    for (int i = 0; i < N; i++)
    {
        std::cin >> host[i];
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    /* na linha abaixo os dados são copiados para GPU */
    thrust::device_vector<double> dev(host);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cerr << "Alocação e cópia para GPU: " << runtime.count() << "ms\n";

    double media = thrust::reduce(dev.begin(), dev.end(), 0.0, thrust::plus<double>()) / N;
    std::cout << "Preço médio: " << media << "\n";

    double mediaUltimoAno = thrust::reduce(dev.begin() + N - 365, dev.end(), 0.0, thrust::plus<double>()) / N;
    std::cout << "Preço médio último ano: " << mediaUltimoAno << "\n";

    double maxVal = thrust::reduce(dev.begin(), dev.end(), 0.0, thrust::maximum<double>());
    std::cout << "Preço máximo: " << maxVal << "\n";

    double maxValUltimoAno = thrust::reduce(dev.begin() + N - 365, dev.end(), 0.0, thrust::maximum<double>());
    std::cout << "Preço máximo último ano: " << maxValUltimoAno << "\n";

    double minVal = thrust::reduce(dev.begin(), dev.end(), maxVal, thrust::minimum<double>());
    std::cout << "Preço mínimo: " << minVal << "\n";

    double minValUltimoAno = thrust::reduce(dev.begin() + N - 365, dev.end(), maxVal, thrust::minimum<double>());
    std::cout << "Preço mínimo último ano: " << minValUltimoAno << "\n";
}
