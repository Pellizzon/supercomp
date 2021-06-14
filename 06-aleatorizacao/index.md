# 06 - Algoritmos Aleatorizados

Um gerador de números pseudo-aleatórios (RNG) é um algoritmo **determinístico** que gera uma sequência de números que **parece** aleatória. Essa frase possui dois termos importantes que precisamos destrinchar:

* **determinístico**: Um *RNG* tipicamente recebe como entrada um inteiro *seed* (que representa uma sequência de bits "aleatória") e gera uma sequência de números baseada no *seed*. Ou seja, o algoritmo é **determinístico** pois gera sempre a mesma sequência para uma determinada entrada (*seed*).
* **parece aleatória**: Se compararmos duas sequências de números, uma gerada por um *RNG* e outra por uma distribuição uniforme de verdade, não conseguimos dizer qual sequência foi gerada pelo *RNG*.

Ou seja, ao escolhermos um *seed* a sequência gerada será sempre a mesma, mesmo se executarmos o programa em outras máquinas. Isso torna a utilização de *RNGs* para experimentos bastante interessante: é possível **reproduzir** os resultados feitos por outros desenvolvedores/cientistas. Para isto é necessário

1. que o programa permita escolher o *seed* da simulação;
1. que o *seed* usado seja publicado junto com os resultados.

!!! question short
    E se quisermos gerar uma sequência diferente a cada execução do programa? Como poderíamos configurar o *seed* para que isto aconteça?

## Iniciando com RNGs

Muitas implementações de *RNGs*  são divididas em duas partes:

1. **engine**/**random state**: algoritmo que gera um inteiro cujos bits formam uma sequência pseudo-aleatória.
1. **distribution**: utiliza os bits acima para retornar números que sigam alguma distribuição estatística (como normal ou uniforme).

!!! question
    A biblioteca padrão de *C++* disponibiliza diversas funções para utilização de *RNG*s (cabeçalho `<random>` - documentação [neste link](http://cplusplus.com/reference/random/)). Se você quisesse sortear números aleatórios inteiros entre `-2` e `5` quais funções usaria?

    ??? details "Resposta"
        ```
        #include <random>

        ...

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(-2,5);
        distribution(generator); // gera número
        ```

!!! question
    E se você quisesse sortear um número real entre `0` e `1`?

    ??? details "Resposta"
        ```
        #include <random>

        ...

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        distribution(generator); // gera número
        ```

Agora que você já consegue gerar números aleatórios, vamos implementar nossa primeira versão de uma heurística aleatorizada.

!!! example
    Adicionaremos a seguinte variação na nossa heurística: a cada passo de seleção temos `25%` de chance de selecionar um objeto aleatório que ainda não foi utilizado. Ou seja, cada passo do algoritmo segue a seguinte regra

    1. Faça um sorteio aleatório
    2. Com probabilidade `75%` pegue o próximo objeto não selecionado de acordo com a heurística (mais leve ou mais caro)
    3. Com probabilidade `25%` selecione um objeto qualquer dos que não foram analisados ainda.

    Note que não mudamos o próximo elemento ao fazer a seleção aleatória. Adote `seed=10` nesta tarefa.

    **Dica**: agora é possível que eu olhe um produto mais de uma vez. Você precisará checar isso no seu programa!

    ??? details "Resposta"
        Os arquivos `in-*.txt` contém entradas para teste. Os arquivos `out-caro-(rand-?)*.txt` contém as saídas esperadas para as heurísticas do mais caro. Note que como estamos falando de uma probabilidade, o sorteio deverá ser feito no intervalo `[0, 1]`.

!!! question short
    Rode a heurística aleatorizada 10 vezes (como fazer isso?) e anote os valores das mochilas obtidas. Em média, é melhor ou pior que a heuristca sem aleatorização?

## Construindo uma solução inteira aleatória

Vamos agora fazer algo mais absurdo: e se criarmos uma solução toda aleatória?

!!! question
    Como você criaria uma solução aleatoriamente?

    ??? details "Resposta"
        Não existe uma resposta certa aqui. Duas soluções são mais comuns:

        1. passando por cada objeto, pegue-o com probabilidade `50%`.
        2. percorra a lista em ordem aleatória, fazendo o mesmo algoritmo do mais caro/leve.

!!! example
    Tente implementar a abordagem 1 da resposta acima.

!!! example
    Tente implementar a abordagem 2 da resposta acima.

!!! question
    Rode ambos programas acima com vários seeds diferentes e anote abaixo os resultados.

!!! question medium
    Anote aqui comentários sobre a qualidade das soluções aleatórias. Considere tanto o valor dos objetos selecionados quanto o peso.

!!! warning
    Iremos discutir esses resultados na próxima aula.
