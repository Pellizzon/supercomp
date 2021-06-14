# 15 - Efeitos Colaterais

Agora que já conseguimos resolver um problema um pouco mais complexo usando duas abordagens diferentes, vamos aumentar um pouco mais a complexidade dos problemas tratados.

No código `pi_recursivo.cpp` tínhamos uma variável global que podia ser eliminada do código mudando a função recursiva. Isso, porém, nem sempre é possível e precisamos lidar com estas situações.

## Um primeiro teste

Vamos iniciar trabalhando com o seguinte trecho de código (arquivo `vetor_insert.cpp`):

```cpp
std::vector<double> vec;
for (int i = 0; i < N; i++) {
	vec.push_back(conta_complexa(i));
}
```

Vamos supor agora que usaremos o seguinte comando para paralelizar o código acima usando OpenMP:

```
#pragma omp parallel for
```

!!! question choice
	A variável `i` é

	- [ ] shared
	- [x] private
	- [ ] firstprivate

!!! question choice
	A variável `vec` é

	- [x] shared
	- [ ] private
	- [ ] firstprivate

!!! question choice
	O código paralelizado rodaria sem dar erros? Os resultados seriam os esperados?

	- [ ] Sim, o `vector` é capaz de gerenciar os acessos simultâneos
	- [ ] O código acima roda sem erros, mas o conteúdo do vetor pode não estar correto ao fim do programa
	- [x] Não, o código acima dá erro ao executar.

	!!! details "Resposta"
		Rode e veja o que acontece ;)

!!! progress
	Clique após rodar o programa

Agora que vimos o que acontece, vamos consertar isso!

!!! danger
	Nosso código dá erro pois a operação `push_back` **modifica o vetor**! Ou seja, é equivalente às operações de escrita que fazíamos nas aulas anteriores (e que também davam errado).

Vamos ver então duas abordagens importantes para contornar esse problema.

## Seções críticas

Antes de começar, vamos aprender mais um aspecto de OpenMP: diretivas para compartilhamento de dados. Já vimos na aula 13 as 3 principais opções:

- `shared`
- `private`
- `firstprivate`

O que não falamos ainda é que podemos forçar a especificação de diretivas de compartilhamento para **todas** as variáveis usadas nas construções `omp parallel`, `omp task` e `omp parallel for`.

!!! tip
	Ao adicionarmos `default(none)` logo após as diretivas acima precisaremos especificar, para cada variável usada, sua diretiva de compartilhamento. Isso torna muito mais fácil identificar casos de compartilhamento indevido de dados.

	A partir desse ponto estaremos supondo que todo código criado usará `default(none)`.

A primeira abordagem usada terá a missão de indicar que um conjunto de linhas contém uma operação que possui efeitos colaterais. Dessa maneira, podemos evitar conflitos se **só permitirmos que essa região rode em uma thread por vez**.

Fazemos isso usando a diretiva `omp critical`:

```cpp
#pragma omp critical
{
	// código aqui dentro roda somente em uma thread por vez.
}
```

Se duas threads chegam ao mesmo tempo no bloco `critical`, uma delas ficará esperando até a outra acabar o bloco. Quando isso ocorrer a thread que esperou poderá prosseguir. Vamos tentar aplicar isso ao código de `vetor_insert.cpp`.

!!! example
	Use `omp critical` para solucionar os problemas de concorrência do código acima.

!!! question short
	Escreva abaixo o tempo que seu código levou para rodar.

!!! progress
	Clique após rodar seu código

Se sua implementação se parecer com o código abaixo, então é bem provável que a versão paralela na verdade tenha demorado o mesmo tempo ou mais que o original.

```cpp
....
#pragma omp parallel for default(none) shared(vec)
for (int i = 0; i < N; i++) {
	#pragma omp critical
	{
		vec.push_back(conta_complexa(i));
	}
}
....
```

!!! question short
	Analise o código novamene e tente explicar por que o programa não ganhou velocidade.

	!!! details "Resposta"
		A operação que produz efeitos colaterais é `vec.push_back`, mas nossa seção crítica envolve também a chamada `conta_complexa(i)`.

!!! example
	Modifique seu código de acordo com a resposta acima. Meça o desempenho e veja que agora há melhora.

Vamos analisar agora a ordem dos dados em `vec`.

!!! question short
	A ordem se mantém igual ao programa sequencial? Você consegue explicar por que?

	!!! details "Resposta"
		Não se mantém. Cada thread chega ao `push_back` em um momento diferente, logo a ordem em que os dados são adicionados no vetor muda.

## Manejo de conflitos usando pré-alocação de memória

Seções críticas são muito úteis quando não conseguimos evitar o compartilhamento de dados. Porém, elas são caras e e feitas especialmente para situações em que região crítica é pequena e chamada um número relativamente pequeno de vezes. 

!!! danger "Como regra, desejamos entrar na região crítica o menor número possível de vezes."

!!! question short
	Reveja o código do início da aula. Seria possível reescrevê-lo para não usar `push_back`?

	!!! details "Resposta"
		Sim, bastaria alocar o vetor com tamanho `N` ao criá-lo. Assim poderíamos atribuir `conta_complexa` direto para a posição de memória desejada.

A estratégia acima é muito importante em alto desempenho e representa uma maneira de evitar seções críticas e sincronização.

!!! done "É sempre melhor alocar memória em blocos grandes antes do paralelismo do que alocar memória frequentemente dentro de regiões paralelas."

Note que fizemos isso na parte de tarefas: ao criarmos variáveis para cada tarefa preencher evitamos a necessidade de usar sincronização.

!!! example
	Modifique o programa para usar a ideia da questão anterior. Meça o desempenho e verifique que tudo funciona normalmente e mais rápido que o original.
