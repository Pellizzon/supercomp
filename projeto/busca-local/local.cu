#include <iostream> // std::cout
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

// nvcc -O3 local.cu -o local-gpu && time ./local-gpu < in8.txt

struct localIter
{
    int M, N, SEED;
    int *itensOwners, *itensValues, *personValue;

    __device__ __host__ int operator()(const int &i)
    {
        // Passo 1: cada objeto é atribuído para uma pessoa aleatória
        thrust::minstd_rand rng(1000 * i * SEED * N);
        thrust::uniform_int_distribution<int> dist(0, M - 1);
        rng.discard(1000);

        for (int k = 0; k < N; k++)
        {
            int personIdx = dist(rng);
            personValue[i * M + personIdx] += itensValues[k];
            itensOwners[i * N + k] = personIdx;
        }

        int P = 0;
        for (int k = 1; k < M; k++)
            if (personValue[i * M + k] < personValue[i * M + P])
                P = k;

        // Passo 3: repetir até não poder mais
        while (1)
        {
            bool donationHappened = false;
            // Passo 2:
            for (int j = 0; j < N; j++)
            {
                // encontra dono atual
                int currentOwner = itensOwners[i * N + j];

                bool canBeDonated = personValue[i * M + currentOwner] - itensValues[j] > personValue[i * M + P];

                // se pode ser doado
                if (canBeDonated)
                {
                    donationHappened = true;

                    //realiza doação
                    personValue[i * M + currentOwner] -= itensValues[j];
                    itensOwners[i * N + j] = P;
                    personValue[i * M + P] += itensValues[j];

                    // encontra nova pessoa P de menor valor (idx P, pessoa com menor valor)
                    P = 0;
                    for (int k = 1; k < M; k++)
                        if (personValue[i * M + k] < personValue[i * M + P])
                            P = k;
                }
            }

            // se nenhuma doação aconteceu, indica que não há mais trocas possíveis
            if (!donationHappened)
                break;
        }

        return personValue[i * M + P];
    }
};

int main()
{
    /* ===============================ENV============================== */
    int DEBUG = getenv("DEBUG") ? std::stoi(getenv("DEBUG")) : 0;
    int SEED = getenv("SEED") ? std::stoi(getenv("SEED")) : 0;
    int ITER = getenv("ITER") ? std::stoi(getenv("ITER")) : 100000;
    /* ================================================================ */

    int N, M;
    std::cin >> N >> M;

    thrust::host_vector<int> itensValues_host(N);

    for (int i = 0; i < N; i++)
        std::cin >> itensValues_host[i];

    thrust::device_vector<int> itensValues(itensValues_host);
    thrust::device_vector<int> personValue(M * ITER);
    thrust::device_vector<int> itensOwners(N * ITER);

    thrust::device_vector<int> answers(ITER);

    localIter f = {
        .M = M,
        .N = N,
        .SEED = SEED + 1,
        .itensOwners = itensOwners.data().get(),
        .itensValues = itensValues.data().get(),
        .personValue = personValue.data().get(),
    };
    thrust::counting_iterator<int> iter(0);
    thrust::transform(iter, iter + answers.size(), answers.begin(), f);

    thrust::host_vector<int> ans_host(ITER);
    thrust::copy(answers.begin(), answers.end(), ans_host.begin());

    int MMS_idx = thrust::max_element(ans_host.begin(), ans_host.end()) - ans_host.begin();

    if (DEBUG)
    {
        for (int i = 0; i < ITER; i++)
        {
            std::cerr << ans_host[i] << " ";
            for (int j = 0; j < N; j++)
                for (int k = 0; k < M; k++)
                    if (itensOwners[j + i * N] == k)
                        std::cerr << k << " ";
            std::cerr << "\n";
        }
    }

    std::cout << ans_host[MMS_idx] << "\n";
    for (int j = 0; j < M; j++)
    {
        for (int i = 0; i < N; i++)
            if (j == itensOwners[i + MMS_idx * N])
                std::cout << i << " ";

        std::cout << "\n";
    }

    return 0;
}