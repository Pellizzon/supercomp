#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <iostream>

// nvcc -std=c++14 -O3 tarefa2.cu -o t2 && ./t2

struct fillRand
{
    thrust::uniform_real_distribution<double> dist;
    thrust::minstd_rand rng;

    fillRand(thrust::uniform_real_distribution<double> dist, thrust::minstd_rand rng) : dist(dist), rng(rng) {}
    __host__ __device__ double operator()(const double &x)
    {
        return dist(rng);
    }
};

int main()
{
    int seed;
    std::cin >> seed;
    // default_random_engine is currently an alias for minstd_rand, and may change in a future version.
    thrust::minstd_rand rng(seed);

    // thrust::uniform_int_distribution<int> dist(-7, 13);
    thrust::uniform_real_distribution<double> dist(25, 40);

    thrust::device_vector<double> vetor(10, 0);

    thrust::transform(vetor.begin(), vetor.end(), vetor.begin(), fillRand(dist, rng));

    // for (auto i = vetor.begin(); i != vetor.end(); i++)
    //     std::cout << *i << " "; // este acesso é lento! -- GPU
    // printf("\n");

    thrust::host_vector<double> host(vetor);

    for (auto i = host.begin(); i != host.end(); i++)
        std::cout << *i << " "; // este acesso é rápido -- CPU
    printf("\n");
}
