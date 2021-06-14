# 05 - Heurísticas

A atividade prática de hoje consiste em implementar heurísticas para a solução do problema da **Mochila binária**.

## Resumo do problema

Dados `N` objetos e uma mochila que comporta até `W` quilos, cada um com peso $w_i$ e valor $v_i$, selecionar objetos com o maior valor possível que caibam dentro da mochila.

**Entrada**:
```
N W
w1 v1
....
wN vN
```

**Saída**:
```
W V opt
o1 ... oT
```

* `W` - peso dos objetos selecionados
* `V` - valor dos objetos selecionados
* `opt`
    * `0` se for usada uma heurística ou busca local
    * `1` se a solução for ótimo global
* `oi` são os índices (*em ordem crescente*) dos objetos selecionados

!!! tip
    Arquivos para verificar a corretude das suas implementações estão disponíveis nesta pasta. Eles estão nomeados como `in-*.txt`, `mais-caro-out-*.txt` e `mais-leve-out-*.txt`.

## Mais caro primeiro

A ideia desta heurística é não deixar nenhum objeto valioso para trás! Por isso vamos ser ganaciosos e pegar **primeiro os objetos mais caros**! Se um objeto valioso não couber passamos para os mais baratos e prosseguimos até examinar todos objetos.

!!! question
    Escreva abaixo um algoritmo em pseudo-código para implementar a heurística descrita acima.

    ??? details "Resposta"
        ```
        ids = // vetor inicializado com ids[i] = i
        ordene os vetores ids, v e w de acordo com o vetor de valores v
        peso = 0
        valor = 0
        resposta = //vetor inicializado com 0
        T = 0 // número de objetos selecionados
        para i=1..N
            se peso + w[i] < W então
                resposta[T] = ids[i]
                peso += w[i]
                valor += v[i]
                T += 1

        print peso, valor, 0
        print resposta[0 .. T]
        ```


!!! question
    Qual é a complexidade computacional deste algoritmo? Ele é a melhor implementação possível?

    ??? details "Resposta"
        Se o algoritmo descrito em sua resposta anterior envolver ordenação, então ele tem complexidade $\mathcal{O}(n\log n)$ e é o melhor possível sim (você consegue explicar por que?). Se você fez um loop duplo que procura pelo maior a cada iteração então seu algoritmo é $\mathcal{O}(n^2)$.

!!! example
    Agora que temos um algoritmo, crie uma implementação do programa acima.

    **Dicas**:

    * C++ já possui um algoritmo de ordenação implementado no cabeçalho [`<algorithm>`](http://cplusplus.com/reference/algorithm/sort/). Use-o.
    * Busque por ordenação indireta para entender como ordenar os três vetores ao mesmo tempo.
    * Pode ser conveniente organizar os dados usando `struct`.

## Mais leve primeiro

Vamos testar uma abordagem oposta: **quantidade agora é o foco**. Por isso vamos ser práticos e pegar **o maior número de objetos possível**! Começaremos agora pelos objetos mais leves e vamos torcer para que a quantidade grande de objetos selecionados resulte em uma mochila com alto valor.

!!! question
    Compare esta heurística com a da seção anterior levando em conta o algoritmo em pseudo-código e sua complexidade computacional.

!!! question
    Quais partes do programa da heurística anterior podem ser aproveitadas para implementar a descrita acima?

!!! example
    Implemente agora a heurística do mais leve. Chame seu programa de `mais_leve`, mantendo também o código do anterior.

## Analisando nossas heurísticas

!!! question
    Crie uma entrada em que a heurística do mais valioso seja muito melhor que a do mais leve. Escreva abaixo as saídas de cada programa.

!!! question
    Crie uma entrada em que a heurística do mais leve seja muito melhor que a do mais valioso. Escreva abaixo as saídas de cada programa.

!!! question
    Com base nas suas respostas acima, em quais situações a heurística do mais valioso é melhor?

!!! question
    Com base nas suas respostas acima, em quais situações a heurística do mais leve é melhor?

