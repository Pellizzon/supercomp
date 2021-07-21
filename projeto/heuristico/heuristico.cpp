#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector

typedef struct
{
    std::vector<int> itensIds;
    int idsIdx;
    int totalValue;
} Person;

typedef struct
{
    int id;
    int value;
} Item;

void mais_valioso(std::vector<Item> &itens)
{
    //ordeno dos mais caros para os mais baratos
    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j) { return i.value > j.value; });
}

int main()
{
    int N, M;
    std::cin >> N >> M;

    std::vector<Person> person(M);
    std::vector<Item> itens(N);

    int item_val;
    for (int i = 0; i < N; i++)
    {
        std::cin >> item_val;
        itens[i].id = i;
        itens[i].value = item_val;
    }

    int maxAmountPerPerson = N / M + 1;
    for (int i = 0; i < M; i++)
    {
        person[i].totalValue = 0;
        person[i].itensIds.resize(maxAmountPerPerson);
        person[i].idsIdx = 0;
    }

    // ordenando do mais valioso para o menos valioso
    mais_valioso(itens);

    int actualPersonIdx = 0;
    int idx;
    for (int i = 0; i < N; i++)
    {
        idx = person[actualPersonIdx].idsIdx;
        person[actualPersonIdx].itensIds[idx] = itens[i].id;
        person[actualPersonIdx].totalValue += itens[i].value;

        person[actualPersonIdx].idsIdx++;
        actualPersonIdx++;
        if (actualPersonIdx == M)
            actualPersonIdx = 0;
    }

    int minValue = person.back().totalValue;

    // Resposta final:
    std::cout << minValue << "\n";
    for (Person i : person)
    {
        // ordenando os ids dos itens que cada pessoa recebeu
        std::sort(i.itensIds.begin(), i.itensIds.begin() + i.idsIdx);
        for (int j = 0; j < i.idsIdx; j++)
            std::cout << i.itensIds[j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";

    return 0;
}
