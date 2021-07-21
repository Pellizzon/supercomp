# Prova Final - SuperComputação


## Regras da prova

* **Duração**: 180 minutos
* Cada questão possui uma pasta onde você deverá colocar suas questões.
* Dúvidas dos enunciados da prova serão resolvidos via chat.
* É permitido consultar o material da disciplina durante a prova (tudo o que estiver no repositório da disciplina e o [no site https://insper.github.io/supercomp](https://insper.github.io/supercomp). Isto inclui suas próprias soluções aos exercícios de sala de aula. **Consultas a outros materias (de outros alunos ou na internet em geral) não são permitidos.**
* É possível também consultar a documentação de C++ nos sites [http://cplusplus.com/](http://cplusplus.com/) e [https://cppreference.com](https://cppreference.com) e da thrust no site [https://thrust.github.io/doc/modules.html](https://thrust.github.io/doc/modules.html)
* É permitido usar papel para rascunhar soluções.
* A prova é individual. Qualquer consulta a outras pessoas durante a prova constitui violação ao Código de Ética do Insper.
* Para entregar a prova, basta criar uma pasta `prova-final` no seu repositório de entregas do projeto. A entrega consiste em copiar o zip da prova para esta pasta e dar commit no pacote inteiro (suas soluções mais arquivos da prova)


## Questão 1 **(5,0)**

O código no arquivo *q1/media_movel.cpp* calcula a média móvel de uma série temporal. Iremos usá-los nas próximas partes dessa questão.

### Parte 1 **(2,0)**

Esta implementação do cálculo da média móvel é paralelizável? Se sim, descreva como você faria uma paralelização em **CPU** deste código. Se não, explique como você a modificaria para que seja paralelizável e explique como faria.

### Parte 2 **(3,0)**

Implemente um programa que calcula a média móvel de uma série usando `thrust`. Coloque seus resultados no arquivo *q2/media_movel.cu*. Seu programa deverá retornar os mesmos resultados do programa sequencial da parte 1.

## Questão 2 **(2,0)**

O código no arquivo *q2/q2.cpp* possui três funções `func1`, `func2` e `func3` que são chamadas em sequência. Foi identificado que elas podem ser feitas paralelamente. Crie uma nova versão deste código chamada *q2/q2-paralelo.cpp* e utilize recursos do OpenMP para paralelizá-lo.

## Questão 3 **(4,0)**

Nesta questão iremos trabalhar com uma nova heurística para o projeto. Esta heurística deverá ser aleatorizada e consiste no seguinte algoritmo

1. ordene os objetos por valor
2. para cada objeto:
    1. com probabilidade `p` atribua-o para a pessoa com a parte de menor valor.
    2. com probabilidade `1-p` atribua-o para uma pessoa qualquer.
3. Repita o processo acima `ITER` vezes.

### Parte 1 **(1,5)**

Faça uma implementação palela em CPU deste algoritmo.

### Parte 2 **(1,5)**

Faça uma implementação palela em GPU deste algoritmo.

### Parte 3 **(1,0)**

Escolha 4 tamanhos de entrada (grandes) e compare os resultados dessa nova heurística com a adotada no projeto. Adote `ITER=100 000` e teste valores de `p={0.25, 0.5, 0.75}`. 
