#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <iostream>

// nvcc -std=c++14 -O3 tarefa6_discard.cu -o t6 && ./t6

struct raw_access
{
    int SEED;
    __device__ __host__ double operator()(const int &i)
    {
        // default_random_engine is currently an alias for minstd_rand, and may change in a future version.
        thrust::minstd_rand rng(SEED);
        thrust::uniform_real_distribution<double> dist(0, 1);
        rng.discard(2 * i);

        double x = dist(rng);
        double y = dist(rng);

        if (x * x + y * y <= 1)
            return 1.0;

        return 0.0;
    }
};

int main()
{

    int N = 100000;
    thrust::device_vector<double> vetor(N);

    thrust::counting_iterator<int> iter(0);
    raw_access ra = {.SEED = 0};
    thrust::transform(iter, iter + vetor.size(), vetor.begin(), ra);
    double sum = thrust::reduce(vetor.begin(), vetor.end(), 0.0, thrust::plus<double>());
    std::cout << sum << "\n";

    double pi = (double)4 * sum / N;
    std::cout << "PI monte carlo: " << pi << "\n";
}
