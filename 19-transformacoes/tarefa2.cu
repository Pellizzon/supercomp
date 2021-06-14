#include "imagem.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>

// nvcc -O3 -std=c++14 tarefa2.cu -o t2 && ./t2
// imagem retirada de https://people.sc.fsu.edu/~jburkardt/data/pgma/pgma.html

struct limiar
{
    __host__ __device__ unsigned char operator()(const unsigned char &p)
    {
        return (p > 127 ? 255 : 0);
    }
};

int main()
{
    Imagem img = Imagem::read("0_baboon.pgm");

    thrust::device_vector<unsigned char> device(img.pixels, img.pixels + img.total_size);
    thrust::device_vector<unsigned char> a(img.total_size);

    thrust::transform(device.begin(), device.end(), a.begin(), limiar());

    thrust::copy(a.begin(), a.end(), img.pixels);
    img.write("2_baboon_limiar.pgm");
}
