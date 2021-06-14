# 10 - Branch and Bound

Vamos começar nossa atividade instrumentando nossa busca exaustiva. Dado que a promessa do nosso algoritmo *Branch and Bound* é evitar chegar até o fim de uma solução parcial que não tem chance de ser ótima, faz todo sentido então contarmos quantas vezes chegamos até o fim.

!!! example
    Vamos adicionar dois contadores ao nosso programa

    1. `num_leaf` conta quantas vezes uma solução completa foi comparada com a melhor possível
    2. `num_copy` conta quantas vezes foi encontrada uma solução melhor que a atual.

!!! question short
    Rode para o exemplo `in150.txt` e anote os valores obtidos abaixo.

## Um bound simples: ignorar peso

Nesta seção implementaremos um *Branch and Bound* simples com a seguinte ideia:

!!! quote "BOUND"
    Complete uma solução parcial incluindo na mochila todos os objetos não selecionados. Isto é equivalente a **relaxar a restrição do peso**.

!!! question short
    Os contadores `num_leaf` e `num_copy` se modificariam ao implementar o *Branch and Bound*? Se sim, quais deles?

    ??? tip "Resposta"
        Somente `num_leaf`, já que deixamos de chegar em folhas que não tem chance de serem ótimos globais. `num_copy` continua igual, já que conta o número de vezes que o melhor foi atualizado.

!!! example
    Implemente no seu código o *Branch and Bound* usando o *Bound* acima. Ou seja, você deverá, ao chegar em um objeto

    1. Checar se a soma da solução atual mais o bound é melhor que o melhor possível.
    2. Se não for retorna
    3. Se for prossegue fazendo a escolha para o objeto atual.

!!! example
    Adicione ao seu programa um contador `num_bounds` que conta o número de vezes em que evitamos de testar uma solução parcial até o fim.

!!! question short
    Teste seu programa novamente com a entrada `in150.txt`. Anote abaixo os contadores e interprete seu resultado.

## Analisando nosso bound

Conseguimos algum ganho de desempenho ao criar o último bound. Vamos agora descobrir se ele é bom mesmo.

!!! question short
Como você mediria a altura em que o bound agiu? Seria melhor cortar mais para cima ou mais para baixo?

O valor `num_bound` não ajuda muito a entender se o bound é bom, já que cortar muito pode significar fazê-lo próximo das folhas (e isto gera ganho pequeno de desempenho).

!!! example
Faça seu programa contar o número de vezes em que o bound é ativado em cada nível da recursão. Mostre esses valores no terminal.

!!! question short
Interprete os resultados acima.

## Implementação eficiente do bound

!!! question short
    O bound *Ignorar peso* depende das escolhas feitas até o momento? Ou seja, se tenho 4 objetos, o bound da solução parcial `(1, 0, -, -)` é igual ou diferente do bound da solução parcial `(1, 1, -, -)`?

!!! question short
    Como você poderia economizar trabalho ao calcular o bound? É possível pré-calcular algo?

!!! example
    Reimplemente seu bound, desta vez pré-calculando tudo antes de iniciar a busca_exaustiva.

!!! question short
    Rode novamente com a entrada `in150.txt` e verifique se houve ganho de tempo de execução.

## Avançado: o quão justo é um bound?

Podemos medir quão justo é um bound verificando a diferença entre seu valor e o valor *real* da melhor solução da subárvore de recursão atual. Ou seja, comparamos nossa estimativa otimista com o que aconteceu de verdade ao examinar todas essas soluções.

!!! example
    Faça seu programa guardar a diferença média entre o valor do bound (que é uma estimativa da qualidade final de uma solução) e o melhor valor encontrado para aquele ramo da recursão.

    **Dica**: você vai precisar retornar o valor da melhor mochila encontrada em cada parte.

!!! question
    Interprete os resultados acima.

