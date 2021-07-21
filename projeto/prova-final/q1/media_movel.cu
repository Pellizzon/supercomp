#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>

// nvcc -O3 media_movel.cu -o t1 && ./t1 < in.txt

struct MediaMovel
{
    double *serie;

    MediaMovel(double *serie) : serie(serie){};

    __host__ __device__ double operator()(const int &i)
    {
        if (i < 6)
            return 0;

        double mean = 0.0;

        for (int j = 0; j < 7; j++)
        {
            mean += serie[j + i - 6] / 7;
        }

        return mean;
    }
};

int main()
{
    thrust::host_vector<double> serie;

    while (std::cin.good())
    {
        double t;
        std::cin >> t;
        serie.push_back(t);
    }

    thrust::device_vector<double> serie_device(serie);

    thrust::device_vector<double> media_movel(serie.size());

    thrust::counting_iterator<int> iter(0);

    MediaMovel mm(serie_device.data().get());

    thrust::transform(iter, iter + serie.size(), media_movel.begin(), mm);

    for (auto i = media_movel.begin(); i != media_movel.end(); i++)
    {
        std::cout << *i << "\n"; // este acesso é lento! -- GPU
    }
}