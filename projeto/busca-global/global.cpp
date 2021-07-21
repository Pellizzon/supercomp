#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector
#include <tuple>
#include <math.h>

// g++ -O3 -Wall global.cpp -o global && ./global < in1.txt

/*  
pergunta 1: dado o item i, escolhe a pessoa j que vai recebê-lo
pergunta 2: M^N solucoes validas

tecnica branch and bound:
calcular o valor que essa pessoa possui em uma determinada iteração.
Se for maior que o valor de corte (soma dos itens/número de pessoas),
significa que essa distribuição já é ruim, então "cortamos" ela.

Ainda, se ordernamos os itens em ordem crescente, temos uma "heuristica"
de distribuir os itens do maior para o menor, chegando na resposta ótima mais 
rapidamente.
*/

typedef struct
{
    int value;
    int owner;
    int id;
} Item;

typedef struct
{
    int mms;
    std::vector<Item> distribution;
} Answer;

int calculateMinValue(std::vector<Item> &itens, int N, int M)
{
    std::vector<int> valuePerPerson(M, 0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (itens[j].owner == i)
                valuePerPerson[i] += itens[j].value;

    int min = *std::min_element(valuePerPerson.begin(), valuePerPerson.end());
    return min;
}

int calculatePersonValue(std::vector<Item> &itens, int N, int personId)
{
    int personValue = 0;
    for (int j = 0; j < N; j++)
        if (itens[j].owner == personId)
            personValue += itens[j].value;

    return personValue;
}

void MMS(std::vector<Item> itens, int N, int M, int i, Answer &best, int &compareValue)
{
    if (i == N)
    {
        int newMms = calculateMinValue(itens, N, M);

        if (best.mms < newMms)
        {
            best.distribution = itens;
            best.mms = newMms;
        }

        return;
    }

    for (int j = 0; j < M; j++)
    {
        if (calculatePersonValue(itens, N, j) > compareValue)
            return;

        itens[i].owner = j;
        MMS(itens, N, M, i + 1, best, compareValue);
    }
}

int main()
{
    int DEBUG = getenv("DEBUG") ? std::stoi(getenv("DEBUG")) : 0;
    int N, M; // numero de itens, numero de pessoas
    std::cin >> N >> M;

    std::vector<Item> itens(N);

    Answer best;
    best.distribution.resize(N);
    best.mms = 0;

    int compareValue = 0;
    for (int i = 0; i < N; i++)
    {
        std::cin >> itens[i].value;
        itens[i].id = i;
        compareValue += itens[i].value;
        itens[i].owner = -1;
    }

    compareValue /= M;

    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j)
              { return i.value > j.value; });

    MMS(itens, N, M, 0, best, compareValue);

    std::sort(best.distribution.begin(), best.distribution.end(), [](const auto &i, const auto &j)
              { return i.id < j.id; });

    std::cout << best.mms << "\n";

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
            if (best.distribution[j].owner == i)
                std::cout << j << " ";
        std::cout << "\n";
    }

    if (DEBUG)
        std::cerr << (long)pow(M, N) << "\n";

    return 0;
}