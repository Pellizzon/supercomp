# 11 - Branch and Bound II

Vimos na expositiva que um bound justo é importante para conseguir ganhos grandes de desempenho no Branch and Bound.

## A mochila fracionária

Nosso relaxamento de restrição permite que um objeto seja incluido em frações. Como vimos na aula, existe um algoritmo simples que dá a solução ótima:

1. ordene os objetos por proporção valor/kg.
1. pegue os objetos nesta ordem até não caber mais na mochila
1. se um objeto não couber inteiro, pegue a maior fração possível

A resposta da *Mochila fracionária* provavelmente não será uma *Mochila binária* viável. Mesmo assim, provamos na aula que ela é um **limitante superior** para a *Mochila binária*.

!!! example
    Implemente a mochila fracionária e execute-a para nossa entrada `in150.txt`. Sua saída deve estar no formato abaixo.

    ```
    valor
    (fracao_do_objeto1) .... (fracao_do_objetoN)
    ```

    Veja o arquivo `out150.txt` para a resposta esperada neste exercício.

## A ordem importa? Estratégia *Best-first*

Antes de incorporar o bound usando mochila fracionária vamos observar o efeito da ordem de análise das soluções. Na nossa primeira tentativa fizemos a busca exaustiva na ordem que os objetos foram apresentados na entrada. Isso resultava nos seguintes indicadores

```
num_leaf 770786731 num_copy 44
```

!!! question short
    Suponha que agora você irá analisar os objetos na ordem usada na *Mochila fracionária*. Você espera que `num_leaf` continue igual, aumente ou diminua?


!!! question short
    Suponha que agora você irá analisar os objetos na ordem usada na *Mochila fracionária*. Você espera que `num_copy` continue igual, aumente ou diminua?

!!! example
    Implemente esta estratégia e verifique se os seus resultados se confirmaram.

    ??? details "Resposta"
        Assim como você deve ter respondido nas duas primeiras questões, `num_leaf` deve continuar igual, mas `num_copy` vai diminuir muito.

## O bound *Mochila fracionária*

Vamos agora juntar tudo e implementar o bound *Mochila fracionária* examinando os objetos na ordem da parte anterior.

!!! question short
    Levando em conta que a ordem anterior fez apenas 3 cópias, qual sua expectativa em relação aos possíveis ganhos de desempenho que implementar o bound *Mochila fracionária* pode trazer?

!!! example
    Implemente o bound.

!!! question short
    Quais os valores de `num_leaf`, `num_copy` e `num_bound` para `in150.txt`?

## Analisando o bound

Vamos agora analisar nossas medidas de efetividade do bound.

!!! question medium
    Compare os valores de `num_bound` para *Ignorar peso* e *Mochila fracionária*. Você consegue tirar conclusões a partir deste valor?

    ??? details "Resposta"
        A dica de que tem algo estranho é que `num_bound` é muito menor. Ou seja, a conclusão de que quanto menor o `num_bound` melhor é claramente problemática, já que contraria a intuição de que um bound teria que estar ocorrendo várias vezes. Por outro lado, o programa é realmente muito mais rápido, então deve haver algo que não está sendo medido corretamente.

!!! example
    Implemente no seu código a contagem de quantas vezes o bound foi feito por nível da recursão.

!!! question medium
    Compare o resultado acima com o obtido no mesmo teste na aula anterior. Agora você consegue explicar a razão do desempenho da nova implementação ter sido tão melhor?

    ??? details "Resposta para a aula anterior"
        ```
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 43 32830 536970 250518 792810 1550819 2119857 2475746 2710403 56370234 16347241 14514859 63885823 22061564 111961043 25164499 79575293 47368277 2373877 0 17059720 2265050 423517```

