# 13 - Paralelismo de dados

Nesta prática iremos usar a contrução `omp parallel for` para tratar casos de paralelismo de dados. 

## Revisão - tarefas

O código *pi-numeric-integration.cpp* calcula o `pi` usando uma técnica chamada integração numérica. 

!!! question short
    Examine o arquivo acima. Onde haveriam oportunidades de paralelização?

    !!! details
        O `for` da linha 13 calcula o pi. Ele roda muitas vezes, então é um bom candidato a paralelismo.

!!! question short
    Suponha que você irá tentar dividir os cálculos do programa em duas partes. Como você faria isso?

    !!! details
        Poderíamos dividir o `for` na metade e fazer cada metade em uma tarefa.

!!! example
    Faça a divisão do cálculo do `pi` em duas tarefas. 
    
    * Sua primeira tarefa deverá guardar seu resultado na variável `res_parte1`. 
    * A segunda tarefa deverá guardar seu resultado na variável `res_parte2`. 
    * Você deverá somar os dois resultados após as tarefas acabarem.

!!! question short
    Meça o tempo do programa paralelizado e compare com o original. Verifique também que os resultados continuam iguais.

## O `for` paralelo

Vamos começar nosso estudo do `for` paralelo executando alguns programas e entendendo como essa construção divide as iterações entre threads. 

!!! question short
    Você consegue predizer o resultado do código abaixo? Se sim, qual seria sua saída? Se não, explique por que. 

    ```c
    #pragma omp parallel for
    for (int i = 0; i < 16; i++) {
        std::cout << "Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    ```

!!! example
    O código acima está no programa *exemplo1.cpp*. Execute-o várias vezes e veja se sua resposta acima é condizente com a realidade.

    ??? details "Resposta"
        Não é possível predizer. No caso acima o loop foi dividido igualmente entre as threads, mas isso é uma decisão do compilador e não temos controle sobre qual será seu comportamento. Isso pode variar de compilador para compilador.

        O comportamento automático funciona bem na maioria das vezes. 

!!! question medium
    Examine o código abaixo e responda.

    ```c
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < 16; i++) {
        std::cout << "Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    ```

    1. Quantos cores, no máximo, serão usados?
    2. Você consegue dizer em qual thread cada iteração rodará?
    3. Você consegue dizer quantas iterações cada thread rodará?
    4. Suponha que a thread 4 iniciou a iteração `i=4`. Ela processará somente essa iteração isoladamente? Se sim, explique por que. Se não, diga até qual valor de `i` ela executará.
    5. As alocações mudam a cada execução do programa?


!!! example
    O código acima está no programa *exemplo2.cpp*. Execute-o várias vezes e veja se sua resposta acima é condizente com a realidade.

!!! question medium
    Examine o código abaixo e responda.

    ```c
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < 16; i++) {
        std::cout << "Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    ```

    1. Quantos cores, no máximo, serão usados?
    2. Você consegue dizer em qual thread cada iteração rodará?
    3. Você consegue dizer quantas iterações cada thread rodará?
    4. As alocações mudam a cada execução do programa?

!!! example
    O código acima está no programa *exemplo3.cpp*. Execute-o várias vezes e veja se sua resposta acima é condizente com a realidade.


