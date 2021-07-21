#include <iostream> // std::cout
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

// nvcc -O3 heuristica.cu -o heuristica-gpu && time ./heuristica-gpu < in8.txt

typedef struct
{
    int id;
    int value;
} Item;

struct localIter
{
    int M, N, SEED;
    int *itensOwners, *personValue;
    Item *itens;
    double proba;

    __device__ __host__ int operator()(const int &i)
    {
        // Passo 1: cada objeto é atribuído para uma pessoa aleatória
        thrust::minstd_rand rng(1000 * i * SEED * N);
        thrust::uniform_real_distribution<double> randP(0.0, 1.0);
        thrust::uniform_int_distribution<int> randomPersonSelector(0, M - 1);
        rng.discard(1000);

        for (int j = 0; j < N; j++)
        {
            int personIdx = 0;
            if (randP(rng) < proba)
            {
                for (int k = 1; k < M; k++)
                    if (personValue[i * M + k] < personValue[i * M + personIdx])
                        personIdx = k;
            }
            else
                personIdx = randomPersonSelector(rng);

            personValue[personIdx + i * M] += itens[j].value;

            itensOwners[itens[j].id + i * N] = personIdx;
        }

        int P = 0;
        for (int k = 1; k < M; k++)
            if (personValue[i * M + k] < personValue[i * M + P])
                P = k;

        return personValue[i * M + P];
    }
};

int main()
{
    double proba = 0.75;
    /* ===============================ENV============================== */
    int DEBUG = getenv("DEBUG") ? std::stoi(getenv("DEBUG")) : 0;
    int SEED = getenv("SEED") ? std::stoi(getenv("SEED")) : 0;
    int ITER = getenv("ITER") ? std::stoi(getenv("ITER")) : 100000;
    /* ================================================================ */

    int N, M;
    std::cin >> N >> M;

    thrust::host_vector<Item> itens_host(N);
    int item_val;
    for (int i = 0; i < N; i++)
    {
        std::cin >> item_val;
        itens_host[i].id = i;
        itens_host[i].value = item_val;
    }

    thrust::sort(itens_host.begin(), itens_host.end(), [](const auto &i, const auto &j)
                 { return i.value > j.value; });

    thrust::device_vector<Item> itens(itens_host);
    thrust::device_vector<int> personValue(M * ITER);
    thrust::device_vector<int> itensOwners(N * ITER);

    thrust::device_vector<int> answers(ITER);

    localIter f = {
        .M = M,
        .N = N,
        .SEED = SEED + 1,
        .itensOwners = itensOwners.data().get(),
        .personValue = personValue.data().get(),
        .itens = itens.data().get(),
        .proba = proba,
    };
    thrust::counting_iterator<int> iter(0);
    thrust::transform(iter, iter + answers.size(), answers.begin(), f);

    thrust::host_vector<int> ans_host(ITER);
    thrust::copy(answers.begin(), answers.end(), ans_host.begin());

    int MMS_idx = thrust::max_element(ans_host.begin(), ans_host.end()) - ans_host.begin();

    // if (DEBUG)
    // {
    //     for (int i = 0; i < ITER; i++)
    //     {
    //         std::cerr << ans_host[i] << " ";
    //         for (int j = 0; j < N; j++)
    //             for (int k = 0; k < M; k++)
    //                 if (itensOwners[j + i * N] == k)
    //                     std::cerr << k << " ";
    //         std::cerr << "\n";
    //     }
    // }

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