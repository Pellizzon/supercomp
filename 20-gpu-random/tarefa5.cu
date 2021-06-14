#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <iostream>

// nvcc -std=c++14 -O3 tarefa5.cu -o t5 && ./t5

struct raw_access
{

    __device__ __host__ double operator()(const int &i)
    {
        thrust::minstd_rand rng;
        thrust::uniform_real_distribution<double> dist(25, 40);
        rng.discard(i);
        return dist(rng);
    }
};

int main()
{

    int N = 10;
    thrust::device_vector<double> vetor(N);

    thrust::counting_iterator<int> iter(0);
    raw_access ra;
    thrust::transform(iter, iter + vetor.size(), vetor.begin(), ra);

    // for (auto i = vetor.begin(); i != vetor.end(); i++)
    //     std::cout << *i << " "; // este acesso é lento! -- GPU
    // printf("\n");

    thrust::host_vector<double> host(vetor);

    for (auto i = host.begin(); i != host.end(); i++)
        std::cout << *i << " "; // este acesso é rápido -- CPU
    printf("\n");
}
