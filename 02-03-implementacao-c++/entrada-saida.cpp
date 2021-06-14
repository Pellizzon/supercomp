#include <iostream>
#include <math.h>
#include <iomanip> // std::setprecision
#include <vector>
using namespace std;

void tarefa1()
{
    int a;
    double b;
    std::cin >> a >> b;
    std::cout << "Saída: " << a << "; " << b << "\n";
}

void tarefa2()
{
    int n;
    std::cin >> n;
    std::cout << (double)n / 2 << "\n";
}

void tarefa3()
{
    int n;
    std::cin >> n;

    double S = 0.0;
    for (int i = 0; i <= n; i++)
    {
        S += 1 / pow(2, i);
    }

    std::cout << std::fixed;
    std::cout << std::setprecision(15) << S << "\n";
}

void tarefa4()
{
    int n;
    std::cin >> n;

    double *values = new double[n];

    double S = 0.0;
    double mi;
    double S2 = 0.0;
    double sigma_squared;

    for (int i = 0; i < n; i++)
    {
        std::cin >> values[i];
        S += values[i];
    }

    mi = S / n;

    for (int i = 0; i < n; i++)
    {
        S2 += pow(values[i] - mi, 2);
    }

    sigma_squared = S2 / n;

    std::cout << std::fixed;
    std::cout << std::setprecision(9) << mi << " " << sigma_squared << "\n";

    delete[] values;
}

void tarefa5()
{
    int n;
    std::cin >> n;

    std::vector<double> values;

    double S = 0.0;
    double mi;
    double S2 = 0.0;
    double sigma_squared;

    double input;
    for (int i = 0; i < n; i++)
    {
        std::cin >> input;
        values.push_back(input);
        S += values[i];
    }

    mi = S / n;

    for (int i = 0; i < n; i++)
    {
        S2 += pow(values[i] - mi, 2);
    }

    sigma_squared = S2 / n;

    std::cout << std::fixed;
    std::cout << std::setprecision(9) << mi << " " << sigma_squared << "\n";
}

void tarefa6()
{
    int n;
    std::cin >> n;

    std::vector<double> x;
    std::vector<double> y;

    double inputX, inputY;

    for (int i = 0; i < n * 2; i++)
    {
        std::cin >> inputX >> inputY;
        x.push_back(inputX);
        y.push_back(inputY);
    }

    std::vector<std::vector<double>> D;

    double deltaX, deltaY;

    for (int i = 0; i < n; i++)
    {
        std::vector<double> a;
        for (int j = 0; j < n; j++)
        {
            deltaX = x[i] - x[j];
            deltaY = y[i] - y[j];
            a.push_back(sqrt(deltaX * deltaX + deltaY * deltaY));
        }
        D.push_back(a);
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << std::fixed;
            std::cout << std::setprecision(2) << D[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    // g++ -Wall -O3 entrada-saida.cpp -o entrada-saida

    // tarefa1();
    // tarefa2();
    // tarefa3();
    // tarefa4(); // python3 t4.py | ./entrada-saida
    // tarefa5();
    // tarefa6(); // ./entrada-saida < t6-in-*.txt
    return 0;
}
