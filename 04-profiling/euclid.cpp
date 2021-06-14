#include <iostream>
#include <vector>
#include <cmath>

typedef std::vector<std::vector<double>> matriz;

// 1. o tempo relativo de execução: 7.59
// 2. o número absoluto de instruções executadas: 4 720 798
// void calcula_distancias(matriz &mat, std::vector<double> &x, std::vector<double> &y)
// {
//     int n = x.size();
//     for (int i = 0; i < n; i++)
//     {
//         std::vector<double> linha;
//         for (int j = 0; j < n; j++)
//         {
//             double dx = x[i] - x[j];
//             double dy = y[i] - y[j];
//             linha.push_back(sqrt(dx * dx + dy * dy));
//         }
//         mat.push_back(linha);
//     }
// }

// 1. o tempo relativo de execução: 6.73
// 2. o número absoluto de instruções executadas:
void calcula_distancias(matriz &mat, std::vector<double> &x, std::vector<double> &y)
{
    double deltaX, deltaY;

    int n = x.size();

    for (int i = 0; i < n; i++)
    {
        std::vector<double> linha;
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            {
                deltaX = x[i] - x[j];
                deltaY = y[i] - y[j];
                linha.push_back(sqrt(deltaX * deltaX + deltaY * deltaY));
            }
            else
            {
                linha.push_back(mat[j][i]);
            }
        }
        mat.push_back(linha);
    }
}

int main()
{
    //g++ -g euclid.cpp -o euclid
    //valgrind --tool=callgrind ./euclid < t6-in-4.txt > out.txt
    //kcachegrind callgrind.out.* // gerado apos o comando anterior

    matriz mat;
    std::vector<double> x, y;
    int n;

    std::cin >> n;
    x.reserve(n);
    y.reserve(n);
    for (int i = 0; i < n; i++)
    {
        double xt, yt;
        std::cin >> xt >> yt;
        x.push_back(xt);
        y.push_back(yt);
    }

    calcula_distancias(mat, x, y);

    for (auto &linha : mat)
    {
        for (double el : linha)
        {
            std::cout << el << " ";
        }
        std::cout << "\n";
    }

    return 0;
}