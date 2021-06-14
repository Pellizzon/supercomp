#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector

/*  Pseudocódigo retornando o valor da mochila ótima
func(P, V, N, C, i) 
    se i=N
        return 0

    semItem = func(P, V, N, C, i+1)
    if (C - P[i] >= 0)
        comItem = mochila(P, V, N, C - P[i], i + 1) + V[i];
        return max(comItem, semItem) 
    
    return semItem
*/

int mochila(std::vector<int> &P, std::vector<int> &V, int N, int C, int i, std::vector<bool> &usados, std::vector<bool> &melhor)
{
    if (i == N)
    {
        // para retornar qual é a mochila ótima
        int valorUsados = 0, valorMelhor = 0;
        for (int i = 0; i < N; i++)
        {
            if (usados[i])
            {
                valorUsados += V[i];
            }
            if (melhor[i])
            {
                valorMelhor += V[i];
            }
        }

        if (valorUsados > valorMelhor)
        {
            melhor = usados;
        }

        return 0;
    }

    usados[i] = false;
    int semItem = mochila(P, V, N, C, i + 1, usados, melhor);

    if (C - P[i] >= 0)
    {
        usados[i] = true;
        int comItem = mochila(P, V, N, C - P[i], i + 1, usados, melhor) + V[i];
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

    std::vector<bool> usados(N, false), melhor(N, false);

    int valorOtimo = mochila(P, V, N, C, 0, usados, melhor);

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

    return 0;
}
