# 17 - Introdução a GPU

Como visto em aula, programação para GPU requer ferramentas especializadas capazes de gerar código que rode parte na CPU (chamada de *host*) e parte na GPU (chamada de *target*). Nesta parte introdutória usaremos a biblioteca `cuda::thrust`. Ela possui um pequeno conjunto de operações otimizadas para GPU e que podem ser customizadas para diversos propósitos.

!!! note "Documentação oficial"
    A documentação oficial da Thrust está disponível no endereço [https://thrust.github.io/doc/modules.html](https://thrust.github.io/doc/modules.html).

Também vamos focar em usar máquinas pré-configuradas. Instruções de instalação local estão disponíveis no *Anexo 1* deste roteiro.

## Compilação para GPU

Para compilar programas para rodar na GPU devemos usar o compilador `nvcc`. Ele identifica quais porções do código deverão ser compiladas para a GPU. O restante do código, que roda exclusivamente na CPU, é passado diretamente para um compilador *C++* regular e um único executável é gerado contendo o código para CPU e chamadas inseridas pelo `nvcc` para invocar as funções que rodam na GPU.

O `nvcc` e todas as bibliotecas que precisamos estão disponíveis no pacote `nvidia-cuda-toolkit` pronto para instalação via *apt*. A versão disponibilizada não é a mais atual, mas tudo funciona de maneira integrada e não é necessário instalar nada manualmente. 

!!! warning "Se você usar as VMs do Insper então não precisa fazer nada. Todas as ferramentas já estão instaladas lá e a VM já vem pronta para uso."

!!! example
    Verifique que sua instalação funciona compilando o arquivo abaixo.

    >$ nvcc -arch=sm_70 -std=c++14 exemplo1-criacao-iteracao.cu -o exemplo1

Se der tudo certo a execução do programa acima deverá gerar um executável `exemplo1` que roda e produz o seguinte resultado.

```
Host vector: 0 0 12 0 35
Device vector 0 0 0 0 35
```

## Compilação para CPU

Se você ainda não tem uma GPU pode usar o suporte da `thrust` para OpenMP nas nossas primeiras aulas.

!!! danger "Todos os trabalhos serão corrigidos usando GPU usando `nvcc`. Esta alternativa é importante somente para as primeiras aulas, em que nem todos terão acesso ainda a uma GPU. Usaremos isso somente para facilitar o primeiro contato, mas essa opção não é válida para avaliações."

1. baixar o código fonte da `thrust` [no github](https://github.com/NVIDIA/thrust).
2. adicionar as seguintes flags no g++
    * `-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP`: diz que a paralelização de `device_vetor` será usando OpenMP
    * `-I/home/....`: o caminho passado será usado na busca por `include`s. Coloque o caminho do repositório da `thrust`
    * `-fopenmp`: já conhecemos este ;)
    * `-x c++`: força a compilação de arquivos `.cu` como código fonte C++

!!! example
    Verifique que sua instalação funciona compilando o arquivo abaixo.

    >$ g++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -I/caminho/para/thrust/  -fopenmp -x c++ exemplo1-criacao-iteracao.cu -o exemplo1-cpu

Se der tudo certo a execução do programa acima deverá gerar um executável `exemplo1-cpu` que roda e produz o seguinte resultado.

```
Host vector: 0 0 12 0 35
Device vector 0 0 0 0 35
```

!!! danger "Nem tudo o que roda usando thrust/OpenMP roda em GPU. Por essa razão, esse recurso será usado somente para testes e nunca para avaliação."

## Transferência de dados

Como visto na expositiva, a CPU e a GPU possuem espaços de endereçamento completamente distintos. Ou seja, a CPU não consegue acessar os dados na memória da GPU e vice-versa. A `thrust` disponibiliza somente um tipo de container (`vector`) e facilita este gerenciamento deixando explícito se ele está alocado na CPU (`host`) ou na GPU (`device`).  A cópia CPU$\leftrightarrow$ GPU é feita implicitamente quando criamos um `device_vector` ou quando usamos a operação de atribuição entre `host_vector` e `device_vector`. Veja o exemplo abaixo:

```cpp
thrust::host_vector<double> vec_cpu(10); // alocado na CPU

vec1[0] = 20;
vec2[1] = 30;

// aloca vetor na GPU e transfere dados CPU->GPU
thrust::device_vector<double> vec_gpu (vec_cpu);

//processa vec_gpu

vec_cpu = vec_gpu; // copia dados GPU -> CPU
```

A `thrust` usa iteradores em todas as suas funções. Pense em um iterador como um ponteiro para os elementos do array. Porém, um iterador é mais esperto: ele guarda também o tipo do vetor original e suporta operações `++` e `*` para qualquer tipo de dado iterado de maneira transparente.

Vetores `thrust` aceitam os métodos `v.begin()` para retornar um iterador para o começo do vetor e `v.end()` para um iterador para o fim (depois do último elemento). Podemos também somar um valor `n` a um iterador. Isto é equivalente a fazer `n` vezes a operação `++`.  Veja abaixo um exemplo de uso das funções `fill` e `sequence` para preencher valores em um vetor de maneira eficiente.

```cpp
thrust::device_vector<int> v(5, 0); // vetor de 5 ints zerado
// v = {0, 0, 0, 0, 0}
thrust::sequence(v.begin(), v.end()); // preenche com 0, 1, 2, ....
// v = {0, 1, 2, 3, 4}
thrust::fill(v.begin(), v.begin()+2, 13); // dois primeiros elementos = 13
// v = {13, 13, 2, 3, 4}
```

!!! question
    Consulte o arquivo *exemplo1-criacao-iteracao.cu* para um exemplo completo de alocação e transferência de dados e do uso de iteradores.

!!! example
    O fluxo de trabalho "normal" de aplicações usando GPU é receber os dados em um vetor na CPU e copiá-los para a GPU para fazer processamentos. Crie um programa que lê uma sequência de `double`s da entrada padrão em um `thrust::host_vector` e os copia para um `thrust::device_vector`. Teste seu programa com o arquivo *stocks-google.txt*, que contém o preço das ações do Google nos últimos 10 anos.

!!! example
    A criação de um `device_vector` é demorada. Meça o tempo que a operação de alocação e cópia demora e imprima na saída de erros. (Use `std::chrono`).

## Operações de redução

Uma operação genérica de *redução* transforma um vetor em um único valor. Exemplos clássicos de operações de redução incluem *soma*, *média* e *mínimo/máximo* de um vetor.

A `thrust` disponibiliza este tipo de operação otimizada em *GPU* usando a função `thrust::reduce`:

```cpp
val = thrust::reduce(iter_comeco, iter_fim, inicial, op);
// iter_comeco: iterador para o começo dos dados
// iter_fim: iterador para o fim dos dados
// inicial: valor inicial
// op: operação a ser feita.
```

Um exemplo de uso de redução para computar o máximo pode ser visto [aqui](http://thrust.github.io/doc/group__reductions_ga5e9cef4919927834bec50fc4829f6e6b.html#ga5e9cef4919927834bec50fc4829f6e6b). A lista completa de funções que podem ser usadas no lugar de `op` pode ser vista [neste link](http://thrust.github.io/doc/group__predefined__function__objects.html).

!!! example
    Continuando o exercício anterior, calcule as seguintes medidas. Não se esqueça de passar o `device_vector` para a sua função `reduce`

    1. O preço médio das ações nos últimos 10 anos.
    1. O preço médio das ações no último ano (365 dias atrás).
    1. O maior e o menor preço da sequência inteira e do último ano.

Você pode consultar todos os tipos de reduções disponíveis no [site da thrust](https://thrust.github.io/doc/group__reductions.html).


## Transformações ponto a ponto

Além de operações de redução também podemos fazer operações ponto a ponto em somente um vetor (como negar todas as componentes ou calcular os quadrados) quanto entre dois vetores (como somar dois vetores componente por componente ou comparar cada elemento com seu correspondente em outro vetor). A `thrust` dá o nome de `transformation` para este tipo de operação.

```cpp
// para operações entre dois vetores iter1 e iter2. resultado armazenado em out
thrust::transform(iter1_comeco, iter1_fim, iter2_comeco, out_comeco, op);
// iter1_comeco: iterador para o começo de iter1
// iter1_fim: iterador para o fim de iter1
// iter2_comeco: iterador para o começo de iter2
// out_comeco: iterador para o começo de out
// op: operação a ser realizada.
```

Um exemplo concreto pode ser visto abaixo. O código completo está em `exemplo2-transform.cu`

```cpp
thrust::device_vector<double> V1(10, 0);
thrust::device_vector<double> V2(10, 0);
thrust::device_vector<double> V3(10, 0);
thrust::device_vector<double> V4(10, 0);
// inicializa V1 e V2 aqui

//soma V1 e V2
thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(), thrust::plus<double>());

// multiplica V1 por 0.5
thrust::transform(V1.begin(), V1.end(),
                  thrust::constant_iterator<double>(0.5),
                  V4.begin(), thrust::multiplies<double>());
```

As operações que foram usadas no `reduce` também podem ser usadas em um `transform`. Não se esqueça de consultar [a lista de operações](http://thrust.github.io/doc/group__predefined__function__objects.html) para fazer este exercício.

!!! example
    Vamos agora trabalhar com o arquivo `stocks2.csv`. Ele contém a série histórica de ações da Apple e da Microsoft. Seu objetivo é calcular a diferença média entre os preços das ações AAPL e MSFT.

    **Dica**: quebre o problema em duas partes. Primeiro calcule a diferença entre os preços e guarde isto em um vetor. Depois compute a média deste vetor.

