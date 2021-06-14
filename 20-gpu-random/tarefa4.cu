#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <iostream>

// nvcc -std=c++14 -O3 tarefa4.cu -o t4 && ./t4

struct raw_access
{
    thrust::uniform_real_distribution<double> dist;
    thrust::minstd_rand rng;

    raw_access(thrust::uniform_real_distribution<double> dist, thrust::minstd_rand rng) : dist(dist), rng(rng) {}

    __device__ __host__ double operator()(const int &i)
    {
        rng.seed(i * 10000);
        return dist(rng);
    }
};

int main()
{
    thrust::minstd_rand rng;

    thrust::uniform_real_distribution<double> dist(25, 40);

    int N = 10;
    thrust::device_vector<double> vetor(N);

    thrust::counting_iterator<int> iter(0);
    raw_access ra(dist, rng);
    thrust::transform(iter, iter + vetor.size(), vetor.begin(), ra);

    // for (auto i = vetor.begin(); i != vetor.end(); i++)
    //     std::cout << *i << " "; // este acesso é lento! -- GPU
    // printf("\n");

    thrust::host_vector<double> host(vetor);

    for (auto i = host.begin(); i != host.end(); i++)
        std::cout << *i << " "; // este acesso é rápido -- CPU
    printf("\n");
}
