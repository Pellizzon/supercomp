#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector
#include <omp.h>
#include <random>


// g++ -Wall -O3 heuristica.cpp -o heuristica-omp -fopenmp && time ./heuristica-omp < in8.txt

typedef struct
{
    int id;
    int value;
} Item;

typedef struct
{
    int MMS;
    std::vector<int> distribution;
} Answer;

void mais_valioso(std::vector<Item> &itens)
{
    //ordeno dos mais caros para os mais baratos
    std::sort(itens.begin(), itens.end(), [](const auto &i, const auto &j)
              { return i.value > j.value; });
}

int main()
{
    double proba = 0.75;

    int SEED = getenv("SEED") ? std::stoi(getenv("SEED")) : 0;
    int ITER = getenv("ITER") ? std::stoi(getenv("ITER")) : 100000;

    int N, M;
    std::cin >> N >> M;

    std::vector<int> itensValues(N);
    std::vector<int> personValue(M, 0);
    std::vector<int> itensOwners(N);

    std::vector<Item> itens(N);

    int item_val;
    for (int i = 0; i < N; i++)
    {
        std::cin >> item_val;
        itens[i].id = i;
        itens[i].value = item_val;
        itensValues[i] = item_val;
    }

    int NUM_THREADS = omp_get_max_threads();
    std::vector<std::default_random_engine> generators(NUM_THREADS);
    for (int j = 0; j < NUM_THREADS; j++)
    {
        std::default_random_engine generator(j * 100 + SEED);
        generators[j] = generator;
    }

    std::uniform_int_distribution<int> randomPersonSelector(0, M - 1);
    std::uniform_real_distribution<double> randP(0.0, 1.0);

    Answer ans;
    ans.distribution.resize(N);
    ans.MMS = 0;

    // ordenando do mais valioso para o menos valioso
    mais_valioso(itens);

   
    #pragma omp parallel for firstprivate(generators, itensValues, personValue, itensOwners, itens)
    for (int i = 0; i < ITER; i++)
    {
        for (auto &i : personValue)
            i = 0;

        for (int j = 0; j < N; j++)
        {
            int personIdx;
            if (randP(generators[omp_get_thread_num()]) < proba)
                personIdx = std::min_element(personValue.begin(), personValue.end()) - personValue.begin();
            else
                personIdx = randomPersonSelector(generators[omp_get_thread_num()]);

            personValue[personIdx] += itens[j].value;

            itensOwners[itens[j].id] = personIdx;
        }

        int P = std::min_element(personValue.begin(), personValue.end()) - personValue.begin();
        if (personValue[P] > ans.MMS)
        {
            #pragma omp critical 
            {
                if (personValue[P] > ans.MMS)
                {
                    ans.MMS = personValue[P];
                    std::copy(itensOwners.begin(), itensOwners.end(), ans.distribution.begin());
                }
            }
        }
    }

    std::cout << ans.MMS << "\n";

    for (int j = 0; j < M; j++)
    {
        for (int i = 0; i < N; i++)
            if (j == ans.distribution[i])
                std::cout << i << " ";

        std::cout << "\n";
    }

    return 0;
}
