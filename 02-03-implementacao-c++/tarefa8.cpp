#include <iostream>
#include <math.h>
#include <iomanip> // setprecision
#include <vector>
using namespace std;

void calcula_distancia(vector<vector<double>> &D, vector<double> x, vector<double> y, int n)
{
    double deltaX, deltaY;

    for (int i = 0; i < n; i++)
    {
        vector<double> a;
        D.push_back(a);
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            {
                deltaX = x[i] - x[j];
                deltaY = y[i] - y[j];
                D[i].push_back(sqrt(deltaX * deltaX + deltaY * deltaY));
            }
            else
            {
                D[i].push_back(D[j][i]);
            }
        }
    }
}

int main()
{
    // g++ -Wall -O3 tarefa8.cpp -o tarefa8
    // time ./tarefa7 < t6-in-4.txt

    int n;
    cin >> n;

    vector<double> x;
    vector<double> y;

    double inputX, inputY;

    for (int i = 0; i < n * 2; i++)
    {
        cin >> inputX >> inputY;
        x.push_back(inputX);
        y.push_back(inputY);
    }

    vector<vector<double>> D;

    calcula_distancia(D, x, y, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << fixed;
            cout << setprecision(2) << D[i][j] << " ";
        }
        cout << "\n";
    }

    return 0;
}