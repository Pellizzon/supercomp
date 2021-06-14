#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector
#include <numeric>   // std:iota

/*  
Branch and Bound II
Organizando por fração
*/

struct Item
{
    int id;
    int weight;
    int value;
    double frac;
    double resposta;
};

void maiorValorPorPeso(std::vector<Item> &itens)
{
    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j) { return i.frac > j.frac; });
}

int main()
{
    // g++ -Wall -O3 tarefa.cpp -o tarefa && ./tarefa < in150.txt

    int n, W;
    std::cin >> n >> W;

    std::vector<Item> itens(n);

    int wi, vi;
    for (int i = 0; i < n; i++)
    {
        std::cin >> wi >> vi;
        itens[i].id = i;
        itens[i].weight = wi;
        itens[i].value = vi;
        itens[i].frac = (double)vi / wi;
    }

    // for (auto &i : itens)
    // {
    //     std::cout << i.weight << " " << i.value << " " << i.frac << "\n";
    // }
    // std::cout << "\n\n";

    maiorValorPorPeso(itens);
    // for (auto &i : itens)
    // {
    //     std::cout << i.weight << " " << i.value << " " << i.frac << "\n";
    // }
    // std::cout << "\n\n";

    double valor = 0, equivalentFrac;
    std::vector<int> resposta(n, 0);
    int espacoRestante = 0, peso = 0;

    for (int i = 0; i < n; i++)
    {
        if (peso + itens[i].weight <= W)
        {
            itens[i].resposta = itens[i].weight / itens[i].weight;
            peso += itens[i].weight;
            valor += itens[i].value;
        }
        else
        {
            espacoRestante = W - peso;
            equivalentFrac = (double)espacoRestante / itens[i].weight;
            peso += espacoRestante;
            valor += (double)equivalentFrac * itens[i].value;
            itens[i].resposta = equivalentFrac;
        }
    }

    std::cout
        << peso << " " << valor << " " << 1 << "\n";

    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j) { return i.id < j.id; });
    for (int i = 0; i < n; i++)
    {
        std::cout << '(' << itens[i].resposta << ')' << " ";
    }
    std::cout << "\n";

    return 0;
}