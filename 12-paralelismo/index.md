# 12 - Introdução a paralelismo

OpenMP é uma tecnologia de computação multi-core usada para paralelizar programas. Sua principal vantagem é oferecer uma transição suave entre código sequencial e código paralelo.

## Primeiros passos

Nesta parte do roteiro usaremos 4 chamadas do OpenMP para criar nossas primeiras threads.

1. `#pragma omp parallel` cria um conjunto de threads. Deve ser aplicado acima de um bloco de código limitado por `{  }`
2. `int omp_get_num_threads();` retorna o número de threads criadas (dentro de uma região paralela)
3. `int omp_get_max_threads();` retorna o número de máximo de threads (fora de uma região paralela)
4. `int omp_get_thread_num();` retorna o id da thread atual (entre 0 e o valor acima, dentro de uma região paralela)

O código abaixo (*exemplo1.cpp*) ilustra como utilizar OpenMP para criar um conjunto de *threads* que rodam em paralelo.

```cpp
#pragma omp parallel
{
    std::cout << "ID:" << omp_get_thread_num() << "/" <<
                        omp_get_num_threads() << "\n";
}
```

Vamos agora fazer alguns experimentos com esse exemplo básico para entender como OpenMP funciona.

!!! example
    Compile o programa de exemplo usando a seguinte linha de comando e rode-o.

    > `$ g++ -O3 exemplo1.cpp -o exemplo1 -fopenmp`

!!! question short
    O OpenMP permite alterar o número máximo de threads criados usando a variável de ambiente `OMP_NUM_THREADS`. Rode `exemplo1` como abaixo.

    > `OMP_NUM_THREADS=2 ./exemplo1`

    Quantas threads foram criadas?

    !!! details
        2 threads.

!!! question short
    Rode agora sem a variável de ambiente. Qual é o valor padrão assumido pelo OpenMP? É uma boa ideia usar mais threads que o valor padrão?

    !!! details
        O valor padrão é o número de threads que o processador pode rodar simultâneamente.

A utilização de `OMP_NUM_THREADS` ajuda a realizar testes de modo a compreender os ganhos de desempenho de um programa conforme mais threads são utilizadas.

Quando uma região paralela inicia são criadas `OMP_NUM_THREADS` *threads* e cada uma roda o bloco de código imediatamente abaixo de maneira independents.

### Escopo de variáveis

Levando em conta o código abaixo, responda as questões abaixo.

```cpp
double res = 0;
#pragma omp parallel
{
    double temp = 10;
    res *= temp;
}
```

!!! question choice
    Quantas cópias da variável `res` existem?

    - [x] 1
    - [ ] 1 para cada thread criada
    - [ ] Nenhuma das anteriores

    !!! details
        Só uma variável `res` existe, pois ela foi declarada fora da região paralela.

!!! question choice
    Quantas cópias da variável `temp` existem?

    - [ ] 1
    - [x] 1 para cada thread criada
    - [ ] Nenhuma das anteriores


    !!! details "Resposta"
        Existem `N` cópias de `temp`, uma criada para cada *thread* existente.


!!! question choice
    Qual o valor de `res` ao final do código abaixo?

    ```cpp
    int res = 1;
    #pragma omp parallel
    {
        for (int i = 0; i < 10000; i++) {
            res += 1;
        }
    }
    ```

    - [ ] 10000
    - [ ] N * 10000
    - [x] Indefinido

!!! example
    Rode o código acima (arquivo *exemplo2.cpp*) e veja se suas expectativas se cumprem. Aproveite e verifique se o programa retorna o mesmo resultado se executado várias vezes.  Chame o professor se você se surpreender com o resultado.

    ??? details "Resposta"
        O código dará resultados estranhos, com `res` não assumindo o valor `N * 10000`. Quanto maior o número de threads mais distante do correto o valor resultante será.

!!! question short
    Mude o limite do `for` para `1000`. Os resultados agora são os esperados? Por que?

    !!! details
        Nesse caso funciona, mas é por acaso. O valor pequeno do `N` faz com que a chance de conflitos diminua, mas ainda é possível. Este tipo de erro "escondido" assim é muito difícil de encontrar. 

!!! danger
    Nos dois exemplos acima as variáveis `res` eram usadas por múltiplas threads! Ou seja, cada thread possui uma **dependência** em relação a `res`. Escrever código sem levar em conta as dependências é um problema que será abordado nas próximas aulas, mas já podemos ver que se duas threads tem uma **dependência de escrita** na mesma variável coisas ruins acontecerão.

### Avançado

!!! question
    Suponha que você tenha um `for` que percorre um vetor `vec` de 1000 elementos e que precisa chamar duas funções `func1`  e `func2` neste vetor. As duas funções não possuem dependências entre si, logo poderiam ser executadas simultâneamente. Como você usaria `omp parallel` para executá-las em paralelo?

    **Dica**: use `omp_get_thread_num()`.

## Paralelismo de tarefas

Vamos agora criar *tarefas* que podem ser executadas em paralelo.

!!! tip "Definição"
    Uma **tarefa** é um bloco de código que é rodado de maneira paralela usando OpenMP. *Tarefas* são agendadas para cada uma das *threads* criadas em um região paralela. Não existe uma associação **1-1** entre *threads* e *tarefas*. Posso ter mais *tarefas* que *threads* e mais *threads* que *tarefas*.

Veja abaixo um exemplo de criação de tarefas.

```cpp
#pragma omp parallel
{
    #pragma omp task
    {
        std::cout << "Estou rodando na tarefa " << omp_get_thread_num() << "\n";
    }
}
std::cout << "eu só rodo quanto TODAS tarefas acabarem.\n";
```

!!! question choice
    O exemplo acima cria quantas tarefas, supondo que `OMP_NUM_THREADS=4`?

    - [ ] 1
    - [x] 4, uma para cada thread
    - [ ] Nenhuma das anteriores


    !!! details
        Como cada thread roda o código da região paralela, cada uma cria exatamente um tarefa.


Para controlar a criação de tarefas em geral usamos a diretiva `master`, que executa somente na thread de índice `0`. Assim conseguimos criar código legível e que deixa bem claro quantas e quais tarefas são criadas.

```cpp
#pragma omp parallel
{
    #pragma omp master
    {
        std::cout << "só roda uma vez na thread:" << omp_get_thread_num() << "\n";
        #pragma omp task
        {
            std::cout << "Estou rodando na thread:" << omp_get_thread_num() << "\n";
        }
    }
}
```

Somente lendo o código acima, responda as questões abaixo.

!!! question choice
    Quantas tarefas são criadas no exemplo acima?

    - [x] 1
    - [ ] N, uma para cada thread
    - [ ] Nenhuma das anteriores

!!! question choice
    A(s) tarefa(s) criada(s) roda(m) em qual thread?

    - [ ] 0
    - [ ] 1
    - [x] Impossível dizer. Em cada execução rodará em uma thread diferente.

!!! example
    Agora roda o código em *exemplo3.cpp* várias vezes e compare suas respostas com a execução do programa.

!!! example
    Complete *exercicio1.cpp* criando duas tarefas. A primeira deverá rodar `funcao1` e a segunda `funcao2`. Salve seus resultados nas variáveis indicadas no código.

!!! question short
    Leia o código e responda. Quanto tempo o código sequencial demora? E o paralelo? Verifique que sua implementação está de acordo com suas expectativas.

    !!! details
        Sequencial demora a soma dos tempos das duas funções. Paralelo demora o tempo da maior delas.
