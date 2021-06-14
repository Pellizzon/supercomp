# 18 - Iteradores em Thrust

Na última aula conseguimos rodar algumas funções em GPU e medir o custo de transferência de dados. Chegamos a conclusão de que, para valer a pena usar a GPU, ou precisamos tratar muitos dados ou precisamos fazer tratamentos pesados.

!!! warning
    Nesta aula trabalharemos extensivamente com a API da `thrust` e pediremos para que vocês consultem a [documentação oficial](https://thrust.github.io/doc/modules.html) da biblioteca. Também pode ser útil ver a [seção de exemplos](https://github.com/NVIDIA/thrust/tree/master/examples) do repositório.

## Iteradores

Continuando a partir do último item da aula anterior, suponha que você deseja calcular a variância das diferenças. Agora que já temos a média, a fórmula da variância é dada por

$$\frac{1}{n}\sum_{i=0}^n (x_i - \mu)^2$$

!!! example
    Já sabemos tudo o que precisamos para fazer este cálculo. Crie um vetor em que todas as componentes sejam iguais a média (usando `thrust::fill`) e compute a variância usando a fórmula acima.

Apesar do código acima funcionar, ele tem um problema grave: é preciso alocar um vetor inteiro somente para preenchê-lo com valores iguais! Se estivermos trabalhando com muitos dados isto significa diminuir bastante a quantidade de dados que conseguimos tratar antes de acabar com a memória da GPU. Podemos contornar isso usando iteradores, que são vetores gerados dinamicamente pela thrust a partir de um valor único ou a partir de outros vetores.

!!! example
    Pesquise a documentação de `thrust::constant_iterator` e use-o para substituir a alocação de memória extra no exercício acima.

Vamos agora nos preparar para a parte 2 criando um programa novamente em cima do arquivo `stocks.txt`.

!!! example
    Leia o arquivo `stocks.txt` e crie um vetor contendo a diferença entre o dia atual e o anterior. Ou seja, dado que o vetor de saída tenha nome `ganho_diario` e o de entrada `stocks`, temos que

    ```ganho_diario[i] = stocks[i+1] - stocks[i]```

    Claramente `ganho_diario.size() == stocks.size() - 1`. Leve isto em conta ao utilizar a operação `transform` para criar o vetor `ganho_diario`.


## Operações customizáveis

Com o vetor `ganho_diario` acima conseguimos saber se o valor da ação subiu ou caiu de um dia para o outro! Duas perguntas se seguem:

1. quantas vezes o valor subiu?
2. qual é o aumento médio, considerando só as vezes em que o valor aumentou de fato?

Podemos implementar essas lógicas usando operações customizadas. Apesar de ser possível escrever essas funções do zero, a `thrust` já possui diversas funções de reduções customizáveis.

!!! warning
    É sempre importante consultar a documentação para entender como essa função será aplicada. Em alguns casos é necessário que a função passada obedeça algumas restrições para que os resultados obtidos façam sentido.

!!! question short
    A `thrust` já tem suporte a operações de redução que são contagens. Veja sua documentação oficial [neste link](https://thrust.github.io/doc/group__counting.html). Qual função dessa página você usaria para contar somente os elementos positivos de `ganhos_diarios`?

!!! warning
    A função `count_if` está atualmente com um erro em sua documentação ([link da issue](https://github.com/thrust/thrust/issues/1148)). Não se esqueça de colocar `const` no parâmetro de `operator()`.

!!! example
    Use a função acima para calcular quantas vezes o valor da ação subiu.

    ??? details "Resposta"
        `1309`

Vamos agora para o segundo item: "Calcular o aumento médio, considerando somente as vezes em que o valor aumentou de fato". Uma estratégia possível é **zerar** todos os elementos negativos do vetor e depois calcular sua soma.

!!! question short
    A [documentação sobre transformações](https://thrust.github.io/doc/group__transformations.html) é bastante vasta. Você consegue encontrar alguma função que possa **substituir** elementos de um vetor baseado em uma condição booleana?

    !!! details
        `replace_if`

!!! example
    Use a função acima para substituir todos os valores negativos por `0` em `ganhos_diarios`.

!!! example
    Calcule agora e média dos valores positivos do vetor. Você já tem todos os que são positivos no exercício acima e a quantidade de valores positivos.

    ??? details "Resposta"
        `5,25179`

