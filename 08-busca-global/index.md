# 08 - Busca exaustiva

## Pseudo-código

Vamos iniciar tentando escrever um algoritmo em pseudo-código para a seguinte ideia:

* Iniciando com o objeto 0:
    * Não inclua ele na mochila: resolva o problema com o restante dos objetos e retorne esse resultado
    * Inclua ele na mochila: resolva o problema com o restante dos objetos e uma mochila de capacidade `C - p[0]`. Retorne o resultado `+ v[0]`.
    * Escolhe a melhor das duas opções acima e retorne.


!!! tip
    Note que pedimos para resolver o problema de novo, mas com menos objetos. Parece que esse é um algoritmo recursivo!

!!! question long
    Escreva um algoritmo recursivo em pseudo-código para resolver o problema da mochila. Seu algoritmo deverá retornar o valor da mochila ótima, mas **NÃO** precisa ainda retornar a mochila que tem esse valor.

!!! question long
    Adapte seu algoritmo acima para, além de retornar a melhor solução, também retornar a mochila que tem esse valor.

    **Dica**: pode ser útil passar um vetor para guardar a melhor solução encontrada.

## Implementação

Vamos agora tentar implementar o algoritmo de busca global que fizemos.

!!! example
    Implemente em C++ seu algoritmo acima.

!!! question
    Teste o seu programa com a entrada `in-aula.txt` (que é a entrada dos slides). Você consegue agora responder à pergunta **Existe mochila com valor maior que 13**?

## Fechamento

!!! question
    Como você avalia os ganhos obtidos pela busca global em relação à busca local?
