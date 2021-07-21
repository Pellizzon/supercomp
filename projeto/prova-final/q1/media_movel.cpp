#include <iostream>
#include <random>

// g++ -O3 media_movel.cpp -o media && ./media < in.txt

int main()
{
    std::vector<double> serie;

    while (std::cin.good())
    {
        double t;
        std::cin >> t;
        serie.push_back(t);
    }

    std::vector<double> media_movel(serie.size());

    std::vector<double> acc;
    for (int i = 0; i < 6; i++)
        acc.push_back(serie[i]);
    for (int i = 6; i < serie.size(); i++)
    {
        acc.push_back(serie[i]);

        for (auto &val : acc)
        {
            media_movel[i] += val / 7;
        }

        acc.erase(acc.begin());
    }

    for (auto &val : media_movel)
    {
        std::cout << val << "\n";
    }
}