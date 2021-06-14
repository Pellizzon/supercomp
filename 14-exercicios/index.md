# 14 - Exercício prático

Agora que já conseguimos resolver problemas simples usando duas abordagens diferentes, vamos aumentar a complexidade dos problemas tratados. Vimos duas abordagens

* `parallel for` - útil para quando precisamos executar a mesma operação em um conjunto grande de dados
* `tasks` - útil para paralelizar tarefas heterogêneas.

Teremos então dois desafios relacionados a paralelizar programas que não são obviamente paralelizáveis.

## Cálculo do `pi` recursivo

Vamos iniciar com um código recursivo para cálculo do pi.

!!! example
    Examine o código em *pi_recursivo.cpp*. Procure entender bem o que está acontecendo antes de prosseguir.

!!! question short
    Onde estão as oportunidades de paralelismo? O código tem dependências?

!!! question medium
    Se o código tiver dependências, é possível refatorá-lo para eliminá-las?

!!! question medium
    Quantas níveis de chamadas recursivas são feitas? Quando o programa para de chamar recursivamente e faz sequencial?

Vamos agora tentar paralelizar o programa usando as duas técnicas.

### Usando `for` paralelo

!!! question short
    Em quais linhas pode haver oportunidade para usar `parallel for`?

!!! example
    Crie uma implementação do *pi_recursivo* usando for paralelo. Meça seu tempo e anote.

!!! example
    O número `MIN_BLK` afeta seu algoritmo? É melhor aumentá-lo ou diminuí-lo? 

!!! question short
    Os ganhos de desempenho foram significativos?

!!! question short
    Como você fez o paralelismo? Precisou definir o número do `for` manualmente ou conseguiu realizar a divisão automaticamente? Comente abaixo sua implementação.


### Usando `task`

Agora vamos usar `task`. Neste caso é vamos adotar a seguinte estratégia: usaremos tarefas para paralelizar as chamadas recursivas feitas em *pi_recursivo.cpp*. 

!!! example
    Crie uma implementação do *pi_recursivo* usando tarefas. Meça seu tempo e anote.
    
    **Dica**: se você precisar esperar tarefas pode usar a diretiva `#pragma omp taskwait`. Ela espera por todas as tarefas criadas pela thread atual.

!!! question short
    Os ganhos de desempenho foram significativos?

!!! question short
    Quantas tarefas foram criadas? Você escolheu essa valor como?

!!! example
    Tente números diferentes de tarefas e verifique se o desempenho melhora ou piora. Anote suas conclusões abaixo. 

### Comparação

!!! question short
    Compare seus resultados das duas abordagens. Anote abaixo seus resultados.

!!! warning
    É possível conseguir tempos muito parecidos com ambas, então se uma delas ficou muito mais lenta é hora de rever o que foi feito.

