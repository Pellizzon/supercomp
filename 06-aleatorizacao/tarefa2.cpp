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
        if (distribution(generator) <= 0.5)
        {
            if (peso + itens[i].weight <= W)
            {
                resposta[T] = itens[i].id;
                peso += itens[i].weight;
                valor += itens[i].value;
                T++;
            }
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
    // g++ -Wall -O3 tarefa2.cpp -o tarefa2 && ./tarefa2 < in1.txt

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

// 98 261 0
// 86 122 205

// 99 310 0
// 36 52 183 230

// 100 213 0
// 111 122 183

// 100 219 0
// 52 183 193

// 97 466 0
// 1 36 86 230 238

// 97 419 0
// 21 122 176 230 242

// 100 458 0
// 1 193 230 238 266

// 100 289 0
// 52 193 230

// 99 439 0
// 21 29 52 176 230

// 100 524 0
// 29 52 86 176 230 242