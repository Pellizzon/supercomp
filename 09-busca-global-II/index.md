# 09 - Comparação de resultados

Já implementamos diversos algoritmos para o problema da mochila binária e chegou a hora de compararmos os resultados por eles obtidos. Nossa ideia aqui é exercitar nossa capacidade de responder perguntas abertas com base em dados.

#### Até qual tamanho de mochila a busca global resolve rápido?

#### O algoritmo de busca local é melhor que as heurísticas?

#### Vale a pena esperar pela busca global? Até que ponto?

## Formulando a pergunta com precisão

Todas as perguntas acima são abertas. Elas admitem diferentes respostas dependendo de nossa interpretação. Nesta seção iremos aprender a reformulá-las de maneira (mais) precisa e a planejar uma série de experimentos que possam apoiar nossa resposta.

!!! question short
    Escolha uma das questões acima para trabalhar nesta questão.

!!! question medium
    Alguns qualificadores comumente usados em discursos informais são considerados "vazios" quanto usados em um contexto mais científico, onde é importante ser preciso nas mensagens. Expressões como

    * "A melhor que B"
    * "rápido, devagar, significativo"
    * "vale a pena"

    são ruins pois não deixam explícito as expectativas de quem as escreveu. Por exemplo, poderíamos ainda perguntar:

    * "A é melhor que B"** sob qual métrica**?
    * **O que é considerado** "rápido, devagar ou significativo"? **1 minuto é rápido ou devagar** (depende da aplicação)
    * **Qual é o critério usado para** "valer a pena"? Tempo? Valor da mochila? O quão próximo do ótimo vale a pena?

    Reescreva a pergunta escolhida agora especificando exatamente o que você gostaria de responder.

!!! warning "Importante"
    Não existe resposta certa para a pergunta acima. Desde que você seja preciso em sua formulação a resposta está correta. Ou seja, **neste momento não estamos questionando se a pergunta faz sentido**, somente **se ela está bem formulada**.

Agora que temos uma pergunta mais precisamente formulada, vamos planejar

!!! question medium
    Temos disponível na aula 08 um gerador de entradas para a mochila. Como você o usaria para gerar dados que te ajudem a responder a pergunta escolhida? Especifique tamanhos de entrada e comente por que você faria estes testes.

!!! question medium
    Com os dados da questão acima em mãos, que ferramentas visuais você usaria para facilitar a comunicação dos resultados?

    * Se sua resposta incluir tabelas, diga o que será mostrado em cada eixo e qual sua interpretação dos dados.
    * Se sua resposta incluir gráficos, explique qual tipo e qual informação você estaria colocando em evidência.

## Implementando de maneira reprodutível

Vamos agora tentar implementar seu plano acima de maneira reprodutível. Ou seja, qualquer pessoa com a infra necessária poderia reexecutar seus experimentos e obter os mesmos dados que você.

!!! example
    Reúna todos os arquivos de entrada usados em uma pasta *in*.

!!! example
    Crie um script python que executa seu programa para todas as entradas acima.

    **Dicas**:

    * reveja a nossa [aula 01](/aulas/01-introducao/) e relembre como criar testes reprodutíveis.

!!! example
    Salve os resultados acima em um dataframe do *Pandas*. Se quiser, salve seus resultados correntes para um arquivo *CSV*.

!!! example
    Crie tabelas ou gráficos a partir do dataframe criado.

!!! question medium
    Usando as tabelas e gráficos criados, responda à pergunta escolhida no início do handout.

