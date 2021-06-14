# 20 - GPU e números aleatórios

## Revisão de números aleatórios

Um gerador de números pseudo-aleatórios (RNG) é um algoritmo **determinístico** que gera uma sequência de números que **parece** aleatória. Essa frase possui dois termos importantes que precisamos destrinchar:

* **determinístico**: Um *RNG* tipicamente recebe como entrada um inteiro *seed* (que representa uma sequência de bits "aleatória") e gera uma sequência de números baseada no *seed*. Ou seja, o algoritmo é **determinístico** pois gera sempre a mesma sequência para uma determinada entrada (*seed*).
* **parece aleatória**: Se compararmos duas sequências de números, uma gerada por um *RNG* e outra por uma distribuição uniforme de verdade, não conseguimos dizer qual distribuição foi gerada pelo *RNG*.

Ou seja, ao escolhermos um *seed* a sequência gerada será sempre a mesma, mesmo se executarmos o programa em outras máquinas. Isso torna a utilização de *RNGs* para experimentos bastante interessante: é possível **reproduzir** os resultados feitos por outros desenvolvedores/cientistas. Para isto é necessário

1. que o programa permita escolher o *seed* da simulação;
1. que o *seed* usado seja publicado junto com os resultados.

Muitas implementações de *RNGs*  são divididas em duas partes:

1. **engine**: algoritmo que gera um inteiro cujos bits formam uma sequência pseudo-aleatória.
1. **distribution**: utiliza os bits acima para retornar números que sigam alguma distribuição estatística (como normal ou uniforme).

A `thrust` contém uma API de geração de números aleatórios muito parecida com a API da biblioteca padrão de C++.

!!! question short
    Consulte a [documentação oficial](https://thrust.github.io/doc/modules.html) da `thrust` e encontre as páginas que descrevem os **engines** e **distributions** implementados.

Vamos agora fazer um uso básico dessas funções.

!!! example
    Crie um programa que leia um inteiro *seed* do terminal e:

    1. crie um objeto `default_random_engine` que o utilize como seed.
    1. mostre no terminal uma sequência de 10 números fracionários tirados de uma distribuição uniforme `[25, 40]`.

    Seu programa deverá estar implementado usando os tipos definidos em `thrust::random`.

Um ponto importante da API `thrust` para geração de números aleatórios é que essas funções podem ser chamadas dentro de operações customizadas! Nosso próximo exercício trata justamente deste uso.

!!! example
    Faça um programa que cria um vetor com 10 elementos e aplica uma operação customizada que seta cada elemento com um valor aleatório. Use as mesmas configurações do exercício anterior. 

!!! warning
    Você pode prosseguir mesmo se seu vetor tem 10 números iguais.

## Gerando números pseudo-aleatórios em GPU

Um desafio em programas paralelos é gerar sequências pseudo-aleatórias de qualidade. Se não tormarmos cuidado acabamos gerando os mesmos números em threads diferentes e desperdiçamos grande quantidade de trabalho!

!!! question short
    Que abordagem você usou para gerar números aleatórios em paralelo com OpenMP na aula `16 - Efeitos Colaterais II`?
    
    !!! details "Resposta"
        Se você realizou todas as tarefas da aula 16, deve ter criado um gerador de números pseudo-aleatórios para cada thread para calcular o valor de $\pi$ usando Monte Carlo.

!!! question short
    Você acha que poderíamos aplicar a mesma abordagem para geração de números aleatórios em GPU? Por quê?

    !!! details "Resposta"
        Sem cuidados adicionais, não obteríamos bons resultados. A abordagem usada na aula 16 pressupõe um número relativamente pequeno de threads, de forma que a sequência criada por cada gerador é menor, porém grande o suficiente para ter qualidade. No caso de usarmos uma GPU, o número de threads pode ser muito grande, de forma que cada gerador geraria sequências muito curtas, e por isso perderíamos em qualidade da sequência. 

## Seeds em programas massivamente paralelos

Em computação massivamente paralela, em geral, existem duas abordagens.

**Abordagem 1**: usar *seeds* diferentes em cada thread.

**Abordagem 2**: usar a mesma *seed* em todas as threads, mas cada uma começa em um ponto diferente da sequência daquela seed.

Note que em ambos os casos os resultados dependem do número de threads usadas! Como vimos em aulas anteriores, um *RNG* tem estado interno e não pode ser facilmente compartilhado entre várias threads.


!!! example
    Implemente a abordagem 1 no exercício da parte anterior. Para isto você pode usar a estratégia de acesso direto aos dados (como foi feito no exercício da imagem) e usar o índice recebido como *seed*.

Você deve ter percebido que todos os números gerados são parecidos, mas não idênticos. Isso ocorre pois geradores com *seed*s próximos geram sequências que são inicialmente parecidas (e depois diferem). Podemos consertar isto usando seeds mais distantes.

!!! example
    Multiplique `i` por um valor grande e tente de novo. Verifique que agora os números são diferentes

!!! example
    Implemente a abordagem 2 no exercício da parte anterior. Para isto você pode *descartar* os *i* primeiros números aleatórios gerados. Procure na documentação oficial como fazer isto.


!!! question short
    Agora compare o desempenho das duas abordagens. Em vez de 10 valores, use `N=100 000`. Há diferença? Existe razão para preferir uma em relação a outra?


## Exercício prático 

Vamos trabalhar com um método probabilístico de estimação do `pi` neste último exercício. O algoritmo sequencial se baseia em sorteios de pontos dentro de um quadrado de lado `2`. Se a distância entre o ponto e o centro do quadrado for menor que 1 então o ponto cai dentro do círculo inscrito no quadrado. A quantidade de pontos que caem dentro do quadrado é proporcional a $\pi$. 

1. `sum = 0`
1. De `i=0` até `N`:
    1. sorteie pontos $x,y \in [0,1]$
    1. se $x^2 + y^2 \leq 1$, `sum += 1`
1. devolva `4 * sum / N`

!!! example 
    Resgate a implementação sequencial deste algoritmo realizada na aula 16 e rode-a para `N=100 000`

!!! example
    Paralelize o código acima em GPU. Use ambas as abordagens acima em programas distintos para lidar com os geradores de números aleatórios.