#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector
#include <numeric>   // std:iota

struct Item
{
    int id;
    int weight;
    int value;
};

void mais_valioso(std::vector<Item> &itens)
{
    //ordeno dos mais caros para os mais baratos
    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j) { return i.value > j.value; });
}

void mais_leve(std::vector<Item> &itens)
{
    //ordeno dos mais leves para os mais pesados
    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j) { return i.weight < j.weight; });
}

int main()
{
    // g++ -Wall -O3 tarefa.cpp -o tarefa
    // ./tarefa < in*.txt

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
    }

    mais_valioso(itens);
    // mais_leve(itens);

    //o algoritmo é o mesmo, o que muda é a ordenação
    int peso = 0, valor = 0, T = 0;
    std::vector<int> resposta(n, 0);
    for (int i = 0; i < n; i++)
    {
        if (peso + itens[i].weight <= W)
        {
            resposta[T] = itens[i].id;
            peso += itens[i].weight;
            valor += itens[i].value;
            T++;
        }
    }

    std::cout << peso << " " << valor << " " << 0 << "\n";

    std::sort(resposta.begin(), resposta.begin() + T);
    for (int i = 0; i < T; i++)
    {
        std::cout << resposta[i] << " ";
    }
    std::cout << "\n";

    return 0;
}