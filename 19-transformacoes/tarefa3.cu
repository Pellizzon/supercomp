#include "imagem.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>

// nvcc -O3 -std=c++14 tarefa3.cu -o t3 && ./t3
// imagem retirada de https://people.sc.fsu.edu/~jburkardt/data/pgma/pgma.html

struct raw_access
{
    unsigned char *ptr;
    int width, heigth;

    raw_access(unsigned char *ptr, int width, int heigth) : ptr(ptr), width(width), heigth(heigth){};

    __device__ __host__ unsigned char operator()(const int &i)
    {
        int x = i % width;
        int y = i / width;

        unsigned char current = ptr[y * width + x];

        unsigned char forward;
        if (x + 1 <= heigth)
            forward = ptr[y * width + (x + 1)];
        else
            forward = 0;

        unsigned char backward;
        if (x - 1 >= 0)
            backward = ptr[y * width + (x - 1)];
        else
            backward = 0;

        unsigned char top;
        if (y - 1 >= 0)
            top = ptr[(y - 1) * width + x];
        else
            top = 0;

        unsigned char down;
        if (y + 1 <= width)
            down = ptr[(y + 1) * width + x];
        else
            down = 0;

        return (current + forward + backward + top + down) / 5;
    }
};

int main()
{
    Imagem img = Imagem::read("0_baboon.pgm");

    thrust::device_vector<unsigned char> device(img.pixels, img.pixels + img.total_size);
    thrust::device_vector<unsigned char> a(img.total_size);

    // std::cout << img.rows << " " << img.cols << "\n";

    thrust::counting_iterator<int> iter(0);
    raw_access ra(device.data().get(), img.cols, img.rows);
    thrust::transform(iter, iter + device.size(), a.begin(), ra);

    thrust::copy(a.begin(), a.end(), img.pixels);
    img.write("1_baboon_media.pgm");
}
