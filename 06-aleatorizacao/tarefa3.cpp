#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector
#include <numeric>   // std:iota
#include <random>
#include <time.h> /* time */

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

void heuristica(int randomSeed, int n, int W, std::vector<Item> itens)
{
    int peso = 0, valor = 0, T = 0;
    std::vector<int> resposta(n, 0);

    std::default_random_engine generator;
    generator.seed(randomSeed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < n; i++)
    {
        // devemos selecionar um item aleatoriamente
        std::uniform_int_distribution<int> randomItemPos(i, n - 1);
        int selectedItemPos = randomItemPos(generator);

        // Coloco o item aleatorio na posicao i e
        // desloco todos os outros itens "para frente"
        Item temp;
        for (int j = selectedItemPos; j > i; j--)
        {
            temp = itens[j];
            itens[j] = itens[j - 1];
            itens[j - 1] = temp;
        }

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
}

int main()
{
    // g++ -Wall -O3 tarefa3.cpp -o tarefa3 && ./tarefa3 < in1.txt

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

    std::default_random_engine generator;
    std::uniform_int_distribution<int> randomSeed(0, 1000000);

    int N = 10; // vezes q o programa irá executar

    for (int i = 0; i < N; i++)
    {
        int newSeed = randomSeed(generator);
        heuristica(newSeed, n, W, itens);
        std::cout << "\n";
    }
    return 0;
}
// 100 242 0
// 3 43 52 106

// 100 286 0
// 26 176 183 230

// 100 118 0
// 16 237 292

// 100 418 0
// 30 78 84 86 176 183 291

// 100 372 0
// 106 176 183 230 240 251

// 100 248 0
// 3 21 176 244

// 100 330 0
// 104 114 176 183 188 193

// 100 133 0
// 11 265

// 100 395 0
// 25 46 68 175 182 183 266

// 100 501 0
// 25 94 176 183 230 242 266