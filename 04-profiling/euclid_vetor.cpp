#include <iostream>
#include <vector>
#include <cmath>

// 1. o tempo relativo de execução: 3.31
// 2. o número absoluto de instruções executadas: 1 945 114
void calcula_distanciasV2(std::vector<double> &mat, std::vector<double> &x, std::vector<double> &y)
{
    int n = x.size();

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            mat[i * n + j] = sqrt(dx * dx + dy * dy);
        }
    }
}

int main()
{
    //g++ -g euclid_vetor.cpp -o euclid_vetor
    //valgrind --tool=callgrind ./euclid_vetor < t6-in-3.txt > out.txt
    //kcachegrind callgrind.out.* // gerado apos o comando anterior

    std::vector<double> mat;
    std::vector<double> x, y;
    int n;

    std::cin >> n;
    x.reserve(n);
    y.reserve(n);

    mat.resize(n * n);
    for (int i = 0; i < n; i++)
    {
        double xt, yt;
        std::cin >> xt >> yt;
        x.push_back(xt);
        y.push_back(yt);
    }

    calcula_distanciasV2(mat, x, y);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}