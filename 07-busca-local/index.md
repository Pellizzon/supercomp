# 07 - Busca local

Nesta aula trabalharemos com um algoritmo chamado "Busca local", que consiste basicamente em fazer pequenas atualizações que melhoram sucessivamente uma solução.

## Solução aleatorizada

Vamos iniciar criando soluções aleatórias. Isto nos permitiria criar uma grande quantidade de soluções e, eventualmente, pegar a melhor delas. Apesar de ser muito mais simples que a busca heurística, a quantidade massiva de soluções geradas tem potencial de encontrar boas soluções.

Vamos trabalhar com um algoritmo bem simples para gerar soluções aleatórias:

* Para cada objeto, selecione-o com probabilidade `0.5`.
* Se o objeto for selecionado, coloque-o na mochile se couber.

!!! question short
    Supondo que só existe uma solução ótima global, qual é a chance de a encontrarmos repetindo o algoritmo acima?

!!! question short
    Supondo que todos os objetos caibam na mochila, quantos são selecionados em média?

!!! example
    Implemente o algoritmo acima. Use `seed=10`.

!!! example
    Repita o algoritmo 10 vezes e pegue somente a melhor solução.

!!! tip
    Use os arquivos de entrada/saída disponibilizados nas aulas passadas.

## Busca local

Vamos agora implementar uma busca local para a Mochila Binária seguindo os dois algoritmos vistos na expositiva.

### Mochila cheia

Para implementar a *Mochila cheia* iremos adotar a seguinte estratégia:

1. Gere uma solução aleatória.
2. Percorre novamente todos os objetos (na ordem da entrada)
3. Se um objeto couber na mochila, inclua-o.

!!! example
    Implemente o algoritmo acima.

!!! example
    Rode a *Mochila cheia* 10 vezes e retorne a melhor solução.

!!! question
    Houve melhoria em relação ao aleatório sozinho? Foi significativa?

<!--
### Substituição de objeto

Para implementar a *Substituição de objeto* iremos adotar a seguinte estratégia:

1. Gere uma solução aleatória.
2. Execute *Mochila Cheia*
3. Para cada objeto (em ordem da entrada):
    1. Verifique, para cada objeto não usado, se uma troca aumentaria o valor da mochila
    2. Em caso positivo, faça a troca e volte para o início do passo 2.
3. Repita enquanto for possível.

!!! example
    Implemente o algoritmo acima.

!!! example
    Rode a *Substituição de objeto* 10 vezes e retorne a melhor solução.

!!! question
    Houve melhoria em relação ao aleatório sozinho? E a *Mochila Cheia*? Foi significativa?
-->

## Fechamento

!!! question
    Como você avalia os ganhos obtidos pela busca local em relação ao aleatório? E em relação a heurística?
