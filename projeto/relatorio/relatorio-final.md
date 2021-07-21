<h1 align="center"> Relatório Final </h1>

<h2 align="center"> Matheus Pellizzon </h2>

<h3 align="center"> Supercomputação - 2021.1 </h3>

## **O problema**

---

O problema `Maximin Share` é um problema *NP-difícil*. Sua resolução visa encontrar a distribuição mais igualitária possível de `N` objetos para `M` pessoas. É possível que determinadas pessoas acabem com mais objetos que outras ou também com maior valor acumulado que outras, uma vez que os objetos são indivisíveis. Esse problema, portanto, visa definir qual o *menor* valor que uma pessoa deveria aceitar. Basicamente, queremos maximizar o valor da pessoa que terá o menor valor. Esse é o chamado MMS.

Para exemplificar o problema, temos 6 objetos e 3 pessoas. Cada objeto possui um valor associado a ele: 20, 11, 9, 13, 14, 37. Uma possível distribuição seria:

- Pessoa 1: itens 37 - valor total: 37;
- Pessoa 2: itens 20, 11 - valor total: 31;
- Pessoa 3: itens 14,  9, 13 - valor total: 36.

Tendo em vista que cada linha representa os objetos que cada pessoa recebeu, o MMS é 31 (pessoa do meio). No entanto, é possível distribuir esses objetos de modo a obter um melhor MMS: 

- Pessoa 1: itens 37 - valor total: 37;
- Pessoa 2: itens 20, 14 - valor total: 34;
- Pessoa 3: itens 13,  11, 9 - valor total: 33.

O MMS equivale a 33 (última pessoa), ou seja, essa distribuição é melhor que a outra, pois é mais igualitária.

## **Entradas e saídas**
---

Por padrão, todas as entradas seguem o formato:

```
N M
v1 ... vN
```

Sendo `N` o número de objetos, `M` o número de pessoas e `vi` o valor do objeto `i`.

O formato da saída desse programa é como segue:

```
MMS
objetos da pessoa 1
...
objetos da M
```

Sendo `MMS` o valor do grupo menos valioso e `objetos da pessoa j` a lista de índices dos objetos que a pessoa `j` recebeu nessa distribuição.

## **Soluções**
--- 

### **Heurística**

Cada pessoa deve receber ao menos `N/M` objetos (arredondado para baixo). Para a realização da distribuição, basta ordenar os itens por valor, decrescente, e atribuí-los para cada pessoa sequencialmente. Ao chegar na última pessoa, é preciso voltar para a primeira e continuar a distribuição até acabar os objetos. Para obter a MMS dessa solução basta verificar o valor total recebido pela última pessoa.

Exemplificando com o exemplo inicial:

- 3 pessoas
- 6 objetos de valores: 20, 11, 9, 13, 14, 37.

Ordenando os objetos por valor: 37, 20, 14, 13, 11, 9.

Distribuição:

- Pessoa 1: itens 37, 13 - valor total: 50;
- Pessoa 2: itens 20, 11 - valor total: 31;
- Pessoa 3: itens 14,  9 - valor total: 23.

Portanto, o MMS para essa solução é 23.

### **Busca Local**

A solução local consiste em algumas etapas. Inicialmente, é preciso distribuir os objetos para as pessoas de forma aleatória. Em seguida, é preciso selecionar a pessoa P com o menor valor total e verficar para cada objeto se ele pode ser doado da pessoa que o tem para a pessoa P. Um objeto pode ser doado se o valor total do doador menos o valor do item doado for maior que o valor total da pessoa P.

Caso um objeto possa ser doado, faça a doação, recalcule o MMS e repita esse processo até não existir mais doações possíveis.

Exemplificando com o exemplo inicial:

- 3 pessoas
- 6 objetos de valores: 20, 11, 9, 13, 14, 37.

Fazer uma distribuição aleatória: 

- Pessoa 1: itens 20, 13 - valor total: 33;
- Pessoa 2: itens  9, 11 - valor total: 20;
- Pessoa 3: itens 14, 37 - valor total: 51.

P é a pessoa 2, pois tem o menor valor total (20). Verificando se algum item pode ser doado para ela:

- Objeto 1: 
    - valor: 20;
    - dono: 1;
    - valor do dono: 33.
    
    Se for doado para P:
        - valor do antigo dono: 13;
        - valor do novo dono: 40.
    
    Como 13 é menor que 40, essa troca não pode ocorrer.

- Objeto 5:
    - valor: 14;
    - dono: 3;
    - valor do dono: 51.
    
    Se for doado para P:
        - valor do antigo dono: 37;
        - valor do novo dono: 34.
    
    Como 37 é maior que 34, essa troca pode ocorrer. 
    
Então o novo MMS é 33, dada a nova distribuição:

- Pessoa 1: itens 20, 13 - valor total: 33;
- Pessoa 2: itens 9, 11, 14 - valor total: 34;
- Pessoa 3: itens 37 - valor total: 37.

As verificações de troca devem ser feitas novamente até não ser possível. Com a distribuição atual não é possível realizar mais trocas, portanto essa seria uma possível resolução local. O próximo passo consiste em repetir todas as etapas algumas vezes e escolher qual resolução apresenta o maior MMS.

### **Busca Local em CPU e GPU**

As etapas da busca local são as mesmas. O paralelismo tem o objetivo de acelerar o processo de geração de resoluções possíveis e escolher a melhor. 

### **Busca exaustiva**

Utilizando recursão, é possível gerar todas as distribuições possíveis dados N itens e M pessoas, obtendo M^N distribuições. Basta escolher como resolução a distribuição em que o MMS é máximo.
    
Para 3 itens de valores 11, 37, 20 e 2 pessoas (representadas por 0 e 1) as seguintes distribuições são possíveis:


<table align="center">
<thead>
<tr>
<th align="center">Item 37</th>
<th align="center">Item 20</th>
<th align="center">Item 11</th>
<th align="center">Valor 0</th>
<th align="center">Valor 1</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">0</td>
<td align="center">0</td>
<td align="center">0</td>
<td align="center">68</td>
<td align="center">0</td>
</tr>
<tr>
<td align="center">0</td>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">57</td>
<td align="center">11</td>
</tr>
<tr>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">0</td>
<td align="center">48</td>
<td align="center">20</td>
</tr>
<tr>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">1</td>
<td align="center">37</td>
<td align="center">31</td>
</tr>
<tr>
<td align="center">1</td>
<td align="center">0</td>
<td align="center">0</td>
<td align="center">11</td>
<td align="center">57</td>
</tr>
<tr>
<td align="center">1</td>
<td align="center">0</td>
<td align="center">1</td>
<td align="center">20</td>
<td align="center">48</td>
</tr>
<tr>
<td align="center">1</td>
<td align="center">1</td>
<td align="center">0</td>
<td align="center">11</td>
<td align="center">57</td>
</tr>
<tr>
<td align="center">1</td>
<td align="center">1</td>
<td align="center">1</td>
<td align="center">0</td>
<td align="center">68</td>
</tr>
</tbody>
</table>

O MMS nessa situação é 31. É possível perceber que existem casos "espelhados", que acabam consumindo recursos desnecessariamente. 

A descrição acima dá a entender que quanto mais itens e mais pessoas mais possibilidades serão processadas e maior tempo será consumido para execução. Para contornar essa desvantagem, é possível determinar um valor de corte para parar a execução caso a distribuição seja "ruim". O método consiste em ordenar os itens por valor (decrescente) e, durate a distribuição, verificar se a pessoa atual possui valor maior que o valor de corte. Caso a condição seja verdade, essa distribuição já irá levar a um cenário não ótimo, então pode ser pulada.

Exemplificando:

Valor de corte: (37 + 20 + 11)/2 = 34: 

<table align="center">
<thead>
<tr>
<th align="center">Item 37</th>
<th align="center">Item 20</th>
<th align="center">Item 11</th>
<th align="center">Valor 0</th>
<th align="center">Valor 1</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">0</td>
<td align="center">0</td>
<td align="center">-</td>
<td align="center">57</td>
<td align="center">0</td>
</tr>
</tbody>
</table>
    
*obs: "-" ainda não distribuído.*
    
Portanto, nem é necessário realizar a distribuição do item 11.

## **Testes**
---

Para avaliar o desempenho das soluções anteriores foram geradas entradas variando `N` (e `M` fixo) e outras com `M` variando (e `N` fixo). Essa decisão foi feita para facilitar a visualização e interpretação dos resultados. Submetendo cada um dos algoritmos aos mesmos testes garante que os resultados finais serão justos em relação ao tempo de execução. Além disso, permitem comparar os MMSs obtidos para cada algortimo, a fim de determinar qual o melhor método para resolução do `Maximin Share`.

Vale ressaltar que os algoritmos de busca local, sejam eles sequencial, paralelo em CPU e GPU, repetem a busca 100000 vezes. Isso ocorre com a finalidade de obter uma solução local que equivale a um ponto de máximo global.

Os testes foram realizados em uma máquina com CPU de 4 *cores* (Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz) e uma GPU com 384 CUDA cores (GeForce 940MX). 

### Testes executados em todas as soluções, com M fixo:
---

<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>N</th>
      <th>M</th>
      <th>MMS_Heuristico</th>
      <th>T_Heuristico</th>
      <th>MMS_Local</th>
      <th>T_Local</th>
      <th>MMS_Global</th>
      <th>T_Global</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>14</td>
      <td>5</td>
      <td>92</td>
      <td>0.007312</td>
      <td>130</td>
      <td>0.059702</td>
      <td>130</td>
      <td>0.236005</td>
    </tr>
    <tr>
      <td>15</td>
      <td>5</td>
      <td>146</td>
      <td>0.019012</td>
      <td>184</td>
      <td>0.058811</td>
      <td>184</td>
      <td>11.148053</td>
    </tr>
    <tr>
      <td>16</td>
      <td>5</td>
      <td>88</td>
      <td>0.016759</td>
      <td>125</td>
      <td>0.070490</td>
      <td>125</td>
      <td>5.194191</td>
    </tr>
    <tr>
      <td>17</td>
      <td>5</td>
      <td>117</td>
      <td>0.018752</td>
      <td>159</td>
      <td>0.083992</td>
      <td>159</td>
      <td>16.863090</td>
    </tr>
    <tr>
      <td>18</td>
      <td>5</td>
      <td>163</td>
      <td>0.015947</td>
      <td>204</td>
      <td>0.075088</td>
      <td>204</td>
      <td>130.027999</td>
    </tr>
  </tbody>
</table>
</div>


Os resultados obtidos para um M fixo podem ser observados na tabela anterior. O que mais chama atenção nesse resultados é o tempo de execução da solução global, que pula de aproximadamente 16.86 segundos para 130.03 segundos. 

A visualização do crescimento do tempo de execução dessa solução pode ser observada no gráfico abaixo. O tempo de execução da global, mesmo utilizando a técnica de *branch and bound*, se comporta como uma exponencial. Caso fosse viável executar testes em maiores quantidades, seria possível observar que a busca do melhor MMS entre muitos cenários, passaria de minutos para horas, de horas para dias, e assim em diante.


<div align="center">
    <img src="relatorio-final_files/relatorio-final_15_0.png">
</div>
    

Também é possível observar as diferentes soluções em relação ao MMS da solução. A busca global possui MMS ótimo, como esperado. A busca local, no entanto, alcança os mesmos valores em tempos muito mais razoáveis.

Como a estratégia delimitada envolve repetir a busca local um determinado número de vezes vezes (100000 vezes, como descrito no início dos testes), a probabilidade de se obter um ponto de máximo local que equivale ao máximo global é alto. Ao mesmo tempo que apresenta MMS ótimo, o tempo e os recursos consumidos pela local não chegam perto da global. Isso pode ser visto principalmente em entradas com muitos itens ou muitas pessoas, onde é viável executar a busca local mas não é viável executar a exaustiva.


<div align="center">
    <img src="relatorio-final_files/relatorio-final_17_0.png">
</div>
    


### Testes executados em todas as soluções, com N fixo:
---


<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>N</th>
      <th>M</th>
      <th>MMS_Heuristico</th>
      <th>T_Heuristico</th>
      <th>MMS_Local</th>
      <th>T_Local</th>
      <th>MMS_Global</th>
      <th>T_Global</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>14</td>
      <td>4</td>
      <td>133</td>
      <td>0.005054</td>
      <td>166</td>
      <td>0.054012</td>
      <td>166</td>
      <td>0.009035</td>
    </tr>
    <tr>
      <td>14</td>
      <td>5</td>
      <td>119</td>
      <td>0.006722</td>
      <td>156</td>
      <td>0.052893</td>
      <td>156</td>
      <td>0.169022</td>
    </tr>
    <tr>
      <td>14</td>
      <td>6</td>
      <td>126</td>
      <td>0.004374</td>
      <td>156</td>
      <td>0.052282</td>
      <td>156</td>
      <td>1.337289</td>
    </tr>
    <tr>
      <td>14</td>
      <td>7</td>
      <td>48</td>
      <td>0.004261</td>
      <td>94</td>
      <td>0.060653</td>
      <td>94</td>
      <td>29.037224</td>
    </tr>
    <tr>
      <td>14</td>
      <td>8</td>
      <td>77</td>
      <td>0.004367</td>
      <td>98</td>
      <td>0.060246</td>
      <td>98</td>
      <td>65.731313</td>
    </tr>
  </tbody>
</table>
</div>



Variando o número de pessoas, é possível observar o mesmo resultado, que já era esperado dada a conclusão anterior. Mesmo utilizando uma estratégia para "cortar" algumas etapas da busca exaustiva, o tempo de execução volta a se comportar, eventualmente, como uma exponencial.  


<div align="center">
    <img src="relatorio-final_files/relatorio-final_22_0.png">
</div>


É possível perceber a diferença entre realizar "cortes" na busca ou não realizá-los. A marca dos 30 segundos é atingida com 7 pessoas e 14 objetos (~ 678.22 bilhões de possibilidades) para o algoritmo que possui a estratégia de *branch and bound* (acima). Já para o algoritmo global que deve passar por todos os cenários (abaixo), os 50 segundos são atingidos com 5 pessoas e 12 itens (~ 244.14 milhões de possibilidades). Esses dados permitem concluir que é possível melhorar a performance da busca global. No entanto, a estratégia delimitada não é suficiente para superar os tempos reduzidos da busca local.

<div align="center">
    <img src="relatorio-final_files/globalSemOtimizacao.png">
</div>

*Gráfico retirado do relatório intermediário, em que o algoritmo da busca global não possuia a estratégia delimitada e deveria percorrer todos os M^N cenários para atingir o melhor MMS.*

O mesmo caso obervado anteriormente para se repete para o MMS também. Os valores de MMS para as entradas geradas são bem próximas entre as diferentes soluções.


<div align="center">
    <img src="relatorio-final_files/relatorio-final_26_0.png">
</div>
    

### Testes executados na solução heurística e local, com M fixo:
---


<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>N</th>
      <th>M</th>
      <th>MMS_Heuristico</th>
      <th>T_Heuristico</th>
      <th>MMS_Local</th>
      <th>T_Local</th>
      <th>MMS_CPU</th>
      <th>T_CPU</th>
      <th>MMS_GPU</th>
      <th>T_GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6</td>
      <td>5</td>
      <td>56</td>
      <td>0.005048</td>
      <td>63</td>
      <td>0.025669</td>
      <td>63</td>
      <td>0.012694</td>
      <td>63</td>
      <td>0.094023</td>
    </tr>
    <tr>
      <td>7</td>
      <td>5</td>
      <td>34</td>
      <td>0.005059</td>
      <td>40</td>
      <td>0.031266</td>
      <td>40</td>
      <td>0.013675</td>
      <td>40</td>
      <td>0.080435</td>
    </tr>
    <tr>
      <td>8</td>
      <td>5</td>
      <td>40</td>
      <td>0.005313</td>
      <td>57</td>
      <td>0.033896</td>
      <td>57</td>
      <td>0.016210</td>
      <td>57</td>
      <td>0.071809</td>
    </tr>
    <tr>
      <td>9</td>
      <td>5</td>
      <td>21</td>
      <td>0.005027</td>
      <td>37</td>
      <td>0.037290</td>
      <td>37</td>
      <td>0.016708</td>
      <td>37</td>
      <td>0.095054</td>
    </tr>
    <tr>
      <td>10</td>
      <td>5</td>
      <td>61</td>
      <td>0.004998</td>
      <td>99</td>
      <td>0.044056</td>
      <td>99</td>
      <td>0.019048</td>
      <td>99</td>
      <td>0.076727</td>
    </tr>
  </tbody>
</table>
</div>



<div align="center">
    <img src="relatorio-final_files/relatorio-final_30_0.png">
</div>

Apesar do tempo de execução da busca local crescer quase que linearmente, enquanto o tempo da heurística permanece praticamente constante, o MMS da local é melhor que o heurístico. 

Retomando os cenários da busca global, demorariam milhares de horas para encontrar a solução ótima para 5 pessoas e 90 itens, por exemplo. A busca local se destaca nesse aspecto; é possível obter uma solução ótima (ou pelo menos bem próxima) que funciona para entradas grandes e é executada em um tempo muito menor que da global.

Além da comparação entre local e heurística anterior, vale ressaltar as melhorias (em relação ao tempo) obtidas nas buscas locais com paralelismo em CPU ou GPU. Como esperado, o MMS é o mesmo, independente da versão do algoritmo (paralela em CPU ou GPU e sem paralelismo). 
    

<div align="center">
    <img src="relatorio-final_files/relatorio-final_30_1.png">
</div>        


Para a paralela em CPU, o tempo de execução diminui com base na quantidade de threads disponíveis. Como a máquina utilizada para a execução desses testes possui 4 *cores*, era esperado que o tempo fosse reduzido em quase 1/4 (no mundo ideal). No entanto, o que realmente é observado é uma redução de aproximadamente 2/3 da local sequencial. Mesmo assim, ainda não é possível igualar ao tempo da heurística, dada a natureza das operações e complexidade do problema.

Outro fato relevante para esses casos remete ao custo fixo de realizar operações em GPU. Existe um overhead para a parte sequencial do programa. Assim, é possível obervar que o desempenho da GPU é pior que a implementação sequencial e paralela em CPU para entradas reduzidas (nesse teste, menos que 100 objetos). Isso leva a conclusão de que não é necessário paralelizar um programa só porque existe essa possibilidade. 

Para entradas suficientemente grandes, o paralelismo em GPU começa a superar ambas as locais, devido a quantidade de *threads* que a GPU possui. No entanto, como a GPU executa a parte paralela do programa em *chunks*, a melhora no tempo de execução é limitada pela quantidade de *chunks* que estarão em execução simultaneamente, além do tempo fixo para alocação de memória, por exemplo. Mesmo assim não é melhor que a heurística (em relação ao tempo de execução).

<div align="center">
    <img src="relatorio-final_files/relatorio-final_32_0.png">
</div>      
        


<div align="center">
    <img src="relatorio-final_files/relatorio-final_32_1.png">
</div>     
    

Ainda em relação ao MMS, no gráfico anterior, não é muito óbvia a diferença entre as duas soluções, dada a escala utilizada. A diferença no MMS obtido pela local é significante, quando comparada com o MMS obtido pela heurística.

### Testes executados na solução heurística e local, com N fixo:
---

<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>N</th>
      <th>M</th>
      <th>MMS_Heuristico</th>
      <th>T_Heuristico</th>
      <th>MMS_Local</th>
      <th>T_Local</th>
      <th>MMS_CPU</th>
      <th>T_CPU</th>
      <th>MMS_GPU</th>
      <th>T_GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>30</td>
      <td>2</td>
      <td>693</td>
      <td>0.005122</td>
      <td>712</td>
      <td>0.074950</td>
      <td>712</td>
      <td>0.031032</td>
      <td>712</td>
      <td>0.104218</td>
    </tr>
    <tr>
      <td>30</td>
      <td>3</td>
      <td>467</td>
      <td>0.004926</td>
      <td>492</td>
      <td>0.080139</td>
      <td>492</td>
      <td>0.037363</td>
      <td>492</td>
      <td>0.122924</td>
    </tr>
    <tr>
      <td>30</td>
      <td>4</td>
      <td>398</td>
      <td>0.004899</td>
      <td>435</td>
      <td>0.091913</td>
      <td>435</td>
      <td>0.042023</td>
      <td>435</td>
      <td>0.115439</td>
    </tr>
    <tr>
      <td>30</td>
      <td>5</td>
      <td>218</td>
      <td>0.004504</td>
      <td>250</td>
      <td>0.104916</td>
      <td>250</td>
      <td>0.064698</td>
      <td>250</td>
      <td>0.136261</td>
    </tr>
    <tr>
      <td>30</td>
      <td>7</td>
      <td>147</td>
      <td>0.004502</td>
      <td>191</td>
      <td>0.120150</td>
      <td>191</td>
      <td>0.073120</td>
      <td>191</td>
      <td>0.136836</td>
    </tr>
  </tbody>
</table>
</div>



Como visto para os testes com M fixo, o tempo que a busca local leva para ser executada cresce com o tamanho das entradas. 

<div align="center">
    <img src="relatorio-final_files/relatorio-final_38_0.png">
</div>     
    
<div align="center">
    <img src="relatorio-final_files/relatorio-final_39_0.png">
</div>     


Nesses testes com N fixo, o mesmo padrão percebido anteriormente se repete. Na maioria dos casos a variação de MMS entre as soluções locais e heurística são relevantes. 

Vale ressaltar que a busca local encontra uma distribuição de "máximo local", por causa da distribuição aleatória dos itens para as pessoas. Novamente, dada a estratégia adotada, é provável chegar a uma resposta ótima para uma distribuição. No entanto ainda é possível que o ponto ótimo não seja encontrado. No caso médio, no entanto, a busca local acaba sendo a melhor solução, quando consideramos tanto o seu tempo de execução (se comparada a global) e a qualidade do resultado (se comparada a heurística).

<div align="center">
    <img src="relatorio-final_files/relatorio-final_41_0.png">
</div>     
    

### Testes executados na solução heurística e local, com itens com valores semelhantes:
---

Por causa do resultado anterior, uma nova análise pode ser realizada.

A busca heurística não é ruim, como aparenta ser dados os testes anteriores. Para casos em que os itens possuem valores muito parecidos, o resultado da heurística é equiparável tanto ao da local quanto da global.

Nos testes seguintes tanto N quanto M foram variados, e os itens possuem valores próximos. 

Nos gráficos, um dos eixos foi fixado como o número de itens para facilitar a visualização dos resultados, visto que o intuito dessa análise é a qualidade das diferentes soluções quanto ao MMS obtido.


<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>N</th>
      <th>M</th>
      <th>MMS_Heuristico</th>
      <th>T_Heuristico</th>
      <th>MMS_Local</th>
      <th>T_Local</th>
      <th>MMS_Global</th>
      <th>T_Global</th>
      <th>MMS_CPU</th>
      <th>T_CPU</th>
      <th>MMS_GPU</th>
      <th>T_GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>5</td>
      <td>95</td>
      <td>0.007352</td>
      <td>95</td>
      <td>0.020313</td>
      <td>95</td>
      <td>0.006079</td>
      <td>95</td>
      <td>0.020206</td>
      <td>95</td>
      <td>0.117534</td>
    </tr>
    <tr>
      <td>6</td>
      <td>5</td>
      <td>95</td>
      <td>0.007180</td>
      <td>97</td>
      <td>0.028957</td>
      <td>97</td>
      <td>0.005253</td>
      <td>97</td>
      <td>0.028748</td>
      <td>97</td>
      <td>0.099971</td>
    </tr>
    <tr>
      <td>7</td>
      <td>5</td>
      <td>96</td>
      <td>0.005382</td>
      <td>100</td>
      <td>0.028733</td>
      <td>100</td>
      <td>0.005766</td>
      <td>100</td>
      <td>0.013552</td>
      <td>100</td>
      <td>0.111922</td>
    </tr>
    <tr>
      <td>8</td>
      <td>5</td>
      <td>97</td>
      <td>0.005341</td>
      <td>100</td>
      <td>0.031724</td>
      <td>100</td>
      <td>0.008856</td>
      <td>100</td>
      <td>0.014226</td>
      <td>100</td>
      <td>0.084448</td>
    </tr>
    <tr>
      <td>9</td>
      <td>5</td>
      <td>99</td>
      <td>0.005446</td>
      <td>100</td>
      <td>0.031996</td>
      <td>100</td>
      <td>0.007143</td>
      <td>100</td>
      <td>0.015029</td>
      <td>100</td>
      <td>0.095767</td>
    </tr>
  </tbody>
</table>
</div>


<div align="center">
    <img src="relatorio-final_files/relatorio-final_46_0.png">
</div>     

    

Nesse caso, há um pulo no valor do MMS simplesmente porque cada pessoa passou a receber pelo menos dois 2 itens. O valor bem próximo, se não igual, entre o MMS das soluções fica em evidência.

<div align="center">
    <img src="relatorio-final_files/relatorio-final_48_0.png">
</div>     

    
<div align="center">
    <img src="relatorio-final_files/relatorio-final_48_1.png">
</div>     
    

Como esperado, o tempo da heurística e da local continuam inferiores ao da global (1° gráfico acima), e o tempo das locais é marginalmente superior ao da heurística (2° gráfico acima). A local em GPU não performa bem nesse teste dado o custo fixo de utilizá-la, característica abordada anteriormente.

## Conclusão
---

Tendo os testes e resultados acima como base, a conclusão que pode ser tirada é que nenhuma solução é melhor que a outra. Tudo depende do cenário (as entradas). Dados cenários diversos, com entradas grandes, que dificilmente seriam executados na global, ou com objetos de valores muito diferentes, que remetem a "fraqueza" da heurística, a busca local sequencial e suas versões em paralelo se destacam. 

Comparando as locais, em específico as paralelas, é possível concluir que tudo depende do cenário. Não adianta paralelizar com GPU uma entrada pequena.
