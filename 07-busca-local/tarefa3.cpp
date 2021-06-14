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

struct Resposta
{
    int weight;
    int value;
    std::vector<int> itens;
};

Resposta local(int randomSeed, int n, int W, std::vector<Item> itens)
{
    int peso = 0, valor = 0, T = 0;
    std::vector<int> resposta(n, 0);

    std::default_random_engine generator;
    generator.seed(randomSeed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    // gera uma solução aleatória
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

    // dada a solução aleatória, percorro os itens, na ordem de entrada,
    // e se algum couber coloco na mochila

    for (int i = 0; i < n; i++)
    {
        // se o item cabe na mochila, tento colocar dentro
        if (peso + itens[i].weight <= W)
        {
            // se o item ainda não está na mochila
            if ((std::find(resposta.begin(), resposta.end(), itens[i].id) == resposta.end()))
            {
                resposta[T] = itens[i].id;
                peso += itens[i].weight;
                valor += itens[i].value;
                T++;
            }
        }
    }

    // std::cout << peso << " " << valor << " " << 0 << "\n";

    // std::sort(resposta.begin(), resposta.begin() + T);
    // for (int i = 0; i < T; i++)
    // {
    //     std::cout << resposta[i] << " ";
    // }
    // std::cout << "\n";

    Resposta resp;
    resp.itens.resize(T);
    resp.value = valor;
    resp.weight = peso;
    for (int i = 0; i < T; i++)
    {
        resp.itens[i] = resposta[i];
    }

    return resp;
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

    std::default_random_engine generator;
    generator.seed(100);
    std::uniform_int_distribution<int> randomSeed(0, 1000000);

    int N = 9; // vezes q o programa irá executar

    Resposta a;
    Resposta temp;
    int newSeed = randomSeed(generator);
    a = local(newSeed, n, W, itens);
    // std::cout << "\n";

    for (int i = 0; i < N; i++)
    {
        int newSeed = randomSeed(generator);
        temp = local(newSeed, n, W, itens);
        if (a.value < temp.value)
            a = temp;
        // std::cout << "\n";
    }

    std::cout << a.weight << " " << a.value << " " << 0 << "\n";
    for (int i = 0; i < (int)a.itens.size(); i++)
    {
        std::cout << a.itens[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
