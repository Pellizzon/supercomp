#include <iostream>  // std::cout
#include <algorithm> // std::sort
#include <vector>    // std::vector
#include <random>
#include <omp.h>

// g++ -Wall -O3 local.cpp -o local && time ./local < in8.txt
// g++ -Wall -O3 local.cpp -o local-omp -fopenmp && time ./local-omp < in8.txt

typedef struct
{
    int MMS;
    std::vector<int> distribution;
} Answer;

int main()
{
    /* ====================ENV=================== */
    int DEBUG = getenv("DEBUG") ? std::stoi(getenv("DEBUG")) : 0;
    int SEED = getenv("SEED") ? std::stoi(getenv("SEED")) : 0;
    int ITER = getenv("ITER") ? std::stoi(getenv("ITER")) : 100000;
    /* ========================================== */

    int N, M;
    std::cin >> N >> M;

    std::vector<int> itensValues(N);
    std::vector<int> personValue(M);
    std::vector<int> itensOwners(N);

    for (int i = 0; i < N; i++)
        std::cin >> itensValues[i];

    #ifdef _OPENMP
        int NUM_THREADS = omp_get_max_threads();
        std::vector<std::default_random_engine> generators(NUM_THREADS);
        for (int j = 0; j < NUM_THREADS; j++)
        {
            std::default_random_engine generator(j * 100 + SEED);
            generators[j] = generator;
        }
    #else
        std::default_random_engine generator(SEED);
    #endif

    std::uniform_int_distribution<int> randomPersonSelector(0, M - 1);

    Answer ans;
    ans.distribution.resize(N);
    ans.MMS = 0;

    #pragma omp parallel for firstprivate(generators, itensValues, personValue, itensOwners)
    for (int t = 0; t < ITER; t++)
    {
        for (auto &i : personValue)
            i = 0;
        // Passo 1: cada objeto é atribuído para uma pessoa aleatória
        for (int i = 0; i < N; i++)
        {
            int personIdx;
            #ifdef _OPENMP
                personIdx = randomPersonSelector(generators[omp_get_thread_num()]);
            #else
                personIdx = randomPersonSelector(generator);
            #endif

            personValue[personIdx] += itensValues[i];
            itensOwners[i] = personIdx;
        }

        int P = std::min_element(personValue.begin(), personValue.end()) - personValue.begin();

        // Passo 3: repetir até não poder mais
        while (1)
        {
            bool donationHappened = false;
            // Passo 2:
            for (int j = 0; j < N; j++)
            {
                // encontra dono atual
                int currentOwner = itensOwners[j];

                bool canBeDonated = personValue[currentOwner] - itensValues[j] > personValue[P];

                // se pode ser doado
                if (canBeDonated)
                {
                    donationHappened = true;

                    //realiza doação
                    personValue[currentOwner] -= itensValues[j];
                    itensOwners[j] = P;
                    personValue[P] += itensValues[j];

                    // encontra nova pessoa P de menor valor (idx P, pessoa com menor valor)
                    P = std::min_element(personValue.begin(), personValue.end()) - personValue.begin();
                }
            }

            // se nenhuma doação aconteceu, indica que não há mais trocas possíveis
            if (!donationHappened)
                break;
        }
        if (DEBUG)
        {
            #pragma omp critical
            {
                std::cerr << personValue[P] << " ";
                for (int j = 0; j < N; j++)
                    for (int i = 0; i < M; i++)
                        if (itensOwners[j] == i)
                            std::cerr << i << " ";
                std::cerr << "\n";
            }
        }

        // truque para evitar complicações devido ao acesso a uma variavel compartilhada entre threads
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

    // para o caso onde há apenas 1 pessoa
    if (personValue.size() == 1)
        ans.MMS = personValue[0];

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