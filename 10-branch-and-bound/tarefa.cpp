#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector

/*  
Branch and Bound
*/

int calculaValor(std::vector<int> &V, std::vector<int> &usados, int &N)
{
    int valor = 0;
    for (int i = 0; i < N; i++)
    {
        if (usados[i])
        {
            valor += V[i];
        }
    }
    return valor;
}

int mochila(
    std::vector<int> &P,
    std::vector<int> &V,
    int N,
    int C,
    int i,
    std::vector<int> &usados,
    std::vector<int> &melhor,
    int &num_leaf,
    int &num_copy,
    std::vector<int> &num_bounds,
    std::vector<int> &aux,
    std::vector<int> &bound)
{
    if (i == N)
    {
        num_leaf++;

        int valorUsados = calculaValor(V, usados, N);
        int valorMelhor = calculaValor(V, melhor, N);

        if (valorUsados > valorMelhor)
        {
            num_copy++;
            melhor = usados;
        }

        return 0;
    }

    aux = usados;

    int valorAux = calculaValor(V, aux, N);
    int valorMelhor = calculaValor(V, melhor, N);

    if (valorAux + bound[i] <= valorMelhor)
    {
        num_bounds[i]++;
        return 0;
    }

    usados[i] = 0;
    int semItem = mochila(P, V, N, C, i + 1, usados, melhor, num_leaf, num_copy, num_bounds, aux, bound);

    if (C - P[i] >= 0)
    {
        usados[i] = 1;
        int comItem = mochila(P, V, N, C - P[i], i + 1, usados, melhor, num_leaf, num_copy, num_bounds, aux, bound) + V[i];
        return std::max(comItem, semItem);
    }

    return semItem;
}

int main()
{
    // g++ -Wall -O3 tarefa.cpp -o tarefa && ./tarefa < in-aula.txt
    int N, C; // numero de objetos, capacidade da mochila
    std::cin >> N >> C;

    std::vector<int> P(N); // peso dos objetos
    std::vector<int> V(N); // valor dos objetos

    int wi, vi;
    for (int i = 0; i < N; i++)
    {
        std::cin >> wi >> vi;
        P[i] = wi;
        V[i] = vi;
    }

    std::vector<int> usados(N, 0), melhor(N, 0), aux(N, 0), num_bounds(N, 0), bound(N, 0);

    int totalValue = 0;
    for (int i : V)
        totalValue += i;
    bound[0] = totalValue;

    for (int i = 1; i < N; i++)
        bound[i] = bound[i - 1] - V[i - 1];

    int num_leaf = 0, num_copy = 0;
    int valorOtimo = mochila(P, V, N, C, 0, usados, melhor, num_leaf, num_copy, num_bounds, aux, bound);

    int wTotal = 0;

    for (int i = 0; i < N; i++)
        if (melhor[i])
            wTotal += P[i];

    std::cout << wTotal << " " << valorOtimo << " "
              << "1"
              << "\n";

    for (int i = 0; i < N; i++)
        if (melhor[i])
            std::cout << i << " ";
    std::cout << "\n";

    std::cout << "Leaf : " << num_leaf << "\n";
    std::cout << "Copy : " << num_copy << "\n";
    for (int i = 0; i < N; i++)
        std::cout << "Recursion " << i << ": " << num_bounds[i] << "\n";

    return 0;
}
