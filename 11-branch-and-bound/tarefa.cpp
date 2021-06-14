#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector

/*  
Branch and Bound
*/

struct Item
{
    int id;
    int weight;
    int value;
    double frac;
    int usados;
    int melhor;
};

void maiorValorPorPeso(std::vector<Item> &itens)
{
    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j) { return i.frac > j.frac; });
}

int calculaValor(std::vector<Item> &I, int &N, bool usados)
{
    int valor = 0;
    for (int i = 0; i < N; i++)
    {
        if (usados)
        {
            if (I[i].usados)
                valor += I[i].value;
        }
        else
        {
            if (I[i].melhor)
                valor += I[i].value;
        }
    }
    return valor;
}

double M_Frac(std::vector<Item> &I, int N, int C, int i)
{
    double value = 0, equivalentFrac;
    int missingSpace = 0, currentWeigth = 0;

    for (int j = i; j < N; j++)
    {
        if (currentWeigth + I[j].weight <= C)
        {
            currentWeigth += I[j].weight;
            value += I[j].value;
        }
        else
        {
            missingSpace = C - currentWeigth;
            equivalentFrac = (double)missingSpace / I[j].weight;
            currentWeigth += missingSpace;
            value += (double)equivalentFrac * I[j].value;
            break;
        }
    }
    return value;
}

int M(std::vector<Item> &I, int N, int C, int i)
{
    if (i == N)
    {

        int valorUsados = calculaValor(I, N, true);
        int valorMelhor = calculaValor(I, N, false);

        if (valorUsados > valorMelhor)
        {
            for (auto &i : I)
                i.melhor = i.usados;
        }

        return 0;
    }

    double bound = (double)calculaValor(I, N, true) + M_Frac(I, N, C, i);

    if (bound <= calculaValor(I, N, false))
        return 0;

    int comItem = 0, semItem;
    if (C - I[i].weight >= 0)
    {
        I[i].usados = 1;
        comItem = M(I, N, C - I[i].weight, i + 1) + I[i].value;
    }
    I[i].usados = 0;
    semItem = M(I, N, C, i + 1);

    return std::max(comItem, semItem);
}

int main()
{
    // g++ -Wall -O3 tarefa.cpp -o tarefa && ./tarefa < in100.txt
    int N, C; // numero de objetos, capacidade da mochila
    std::cin >> N >> C;

    std::vector<Item> I(N);

    int wi, vi;
    for (int i = 0; i < N; i++)
    {
        std::cin >> wi >> vi;
        I[i].id = i;
        I[i].weight = wi;
        I[i].value = vi;
        I[i].frac = (double)vi / wi;
        I[i].usados = 0;
        I[i].melhor = 0;
    }

    maiorValorPorPeso(I);

    int valorOtimo = M(I, N, C, 0);

    int wTotal = 0;

    for (int i = 0; i < N; i++)
        if (I[i].melhor)
            wTotal += I[i].weight;

    std::cout << wTotal << " " << valorOtimo << " "
              << "1"
              << "\n";

    std::sort(I.begin(), I.end(), [](const auto &i, const auto &j) { return i.id < j.id; });
    for (int i = 0; i < N; i++)
        if (I[i].melhor)
            std::cout << i << " ";
    std::cout << "\n";

    return 0;
}
