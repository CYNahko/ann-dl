---
exercise: mlp
ai_use: "Utilizado para explicar os enunciados e os conceitos de MLP, conferir as contas do Exercício 1, verificar os codigos e os textos escritos"
---

# 3. MLP

!!! abstract "Enunciado"

    [Exercises → MLP](https://insper.github.io/ann-dl/pt/2026.2/exercises/mlp/){:target='_blank'}

## Exercise 1

### Abordagem

Foi considerada uma rede com 2 entradas, 1 camada oculta de 2 neurônios e 1 neurônio de saída, com \(\tanh\) nas duas camadas e perda quadrática sobre uma única amostra (\(N = 1\)). Um passo completo de retropropagação foi feito à mão: passagem direta, cálculo dos gradientes pela regra da cadeia e atualização dos parâmetros por gradiente descendente com \(\eta = 0.3\). Todos os valores foram conferidos com um script em NumPy.

### Dados

\[
\mathbf{x} = \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix},\quad y = 1.0,\quad
\mathbf{W}^{(1)} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix},\quad
\mathbf{b}^{(1)} = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}
\]

\[
\mathbf{W}^{(2)} = \begin{bmatrix} 0.5 & -0.3 \end{bmatrix},\quad b^{(2)} = 0.2,\quad \eta = 0.3
\]

### A - Passagem direta

**1. Pré-ativações ocultas**

\[
\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}
\]

\[
z^{(1)}_1 = 0.3(0.5) + (-0.1)(-0.2) + 0.1 = 0.2700
\]

\[
z^{(1)}_2 = 0.2(0.5) + 0.4(-0.2) - 0.2 = -0.1800
\]

**2. Ativações ocultas**

\[
\mathbf{h}^{(1)} = \tanh(\mathbf{z}^{(1)}) = \begin{bmatrix} \tanh(0.27) \\ \tanh(-0.18) \end{bmatrix} = \begin{bmatrix} 0.2636 \\ -0.1781 \end{bmatrix}
\]

**3. Pré-ativação de saída**

\[
u^{(2)} = \mathbf{W}^{(2)}\mathbf{h}^{(1)} + b^{(2)} = 0.5(0.2636) + (-0.3)(-0.1781) + 0.2 = 0.3852
\]

**4. Saída e perda**

\[
\hat{y} = \tanh(0.3852) = 0.3672
\]

\[
L = (y - \hat{y})^2 = (1 - 0.3672)^2 = 0.4004
\]

### B - Passagem reversa

\[
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -2(1 - 0.3672) = -1.2655
\]

**1. Gradiente na pré-ativação de saída**

Usando \(\tanh'(u) = 1 - \tanh^2(u) = 1 - \hat{y}^2\):

\[
\frac{\partial L}{\partial u^{(2)}} = \frac{\partial L}{\partial \hat{y}}\,(1 - \hat{y}^2) = -1.2655\,(1 - 0.3672^2) = -1.2655 \cdot 0.8651 = -1.0948
\]

**2. Gradientes da camada de saída**

\[
\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \frac{\partial L}{\partial u^{(2)}}\,\mathbf{h}^{(1)\top} = -1.0948\begin{bmatrix} 0.2636 & -0.1781 \end{bmatrix} = \begin{bmatrix} -0.2886 & 0.1950 \end{bmatrix}
\]

\[
\frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial u^{(2)}} = -1.0948
\]

**3. Propagando para a camada oculta**

\[
\frac{\partial L}{\partial \mathbf{h}^{(1)}} = \mathbf{W}^{(2)\top}\,\frac{\partial L}{\partial u^{(2)}} = \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix}(-1.0948) = \begin{bmatrix} -0.5474 \\ 0.3284 \end{bmatrix}
\]

\[
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{h}^{(1)}} \odot \left(1 - (\mathbf{h}^{(1)})^2\right) = \begin{bmatrix} -0.5474 \cdot 0.9305 \\ 0.3284 \cdot 0.9683 \end{bmatrix} = \begin{bmatrix} -0.5094 \\ 0.3180 \end{bmatrix}
\]

**4. Gradientes da camada oculta**

\[
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}}\,\mathbf{x}^\top = \begin{bmatrix} -0.5094 \\ 0.3180 \end{bmatrix}\begin{bmatrix} 0.5 & -0.2 \end{bmatrix} = \begin{bmatrix} -0.2547 & 0.1019 \\ 0.1590 & -0.0636 \end{bmatrix}
\]

\[
\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \begin{bmatrix} -0.5094 \\ 0.3180 \end{bmatrix}
\]

### C - Atualização dos parâmetros

Regra: \(\theta \leftarrow \theta - \eta\,\dfrac{\partial L}{\partial \theta}\), com \(\eta = 0.3\).

| Parâmetro | Antes | Gradiente | Depois |
|---|---|---|---|
| \(W^{(1)}_{11}\) | 0.3000 | -0.2547 | 0.3764 |
| \(W^{(1)}_{12}\) | -0.1000 | 0.1019 | -0.1306 |
| \(W^{(1)}_{21}\) | 0.2000 | 0.1590 | 0.1523 |
| \(W^{(1)}_{22}\) | 0.4000 | -0.0636 | 0.4191 |
| \(b^{(1)}_1\) | 0.1000 | -0.5094 | 0.2528 |
| \(b^{(1)}_2\) | -0.2000 | 0.3180 | -0.2954 |
| \(W^{(2)}_1\) | 0.5000 | -0.2886 | 0.5866 |
| \(W^{(2)}_2\) | -0.3000 | 0.1950 | -0.3585 |
| \(b^{(2)}\) | 0.2000 | -1.0948 | 0.5284 |

**A perda sobe ou desce?** Desce. Cada parâmetro foi movido no sentido oposto ao seu gradiente, que é a direção em que a perda diminui localmente. Como o passo \(\eta = 0.3\) é pequeno o suficiente, a aproximação linear vale e a perda cai. Refazendo a passagem direta com os novos parâmetros, \(\hat{y} = 0.7112\) e \(L = 0.0834\), contra \(0.4004\) antes.

### Código

``` { .python .copy .select linenums='1' title="docs/exercises/mlp/code/ex1_backprop.py" }
--8<-- "docs/exercises/mlp/code/ex1_backprop.py"
```

## Exercise 2

### Abordagem

Foram gerados 1000 pontos em três grupos: a Classe 0 no centro e a Classe 1 dividida em dois grupos em cantos opostos. Os dados foram separados em 80% para treino e 20% para teste. Primeiro, uma regressão logística foi usada como linha de base linear. Em seguida, foi treinado um MLP implementado do zero em NumPy, com passagem direta, perda, retropropagação e atualização dos parâmetros escritas à mão. A classe recebe os tamanhos das camadas como argumento, para ser reaproveitada nos Exercícios 3 e 4.

### A - Gere os dados

Os dados foram gerados com `make_blobs`, usando centros \((0, 0)\), \((3, 3)\) e \((-3, -3)\), com 500, 250 e 250 pontos, respectivamente, desvio padrão \(1.2\) e `random_state=42`. O primeiro grupo forma a Classe 0, e os outros dois formam a Classe 1 (\(y = 1\) se o grupo for diferente de 0). A divisão treino/teste usou `train_test_split` com `test_size=0.2`, `random_state=42` e `stratify=y`, resultando em 800 pontos de treino e 200 de teste.

![Dispersão dos 1000 pontos](figures/fig1_dispersao.png)
/// caption
**Figura 1** — Dispersão dos 1000 pontos, coloridos pela classe.
///

### B - Estabeleça a linha de base

A linha de base foi uma regressão logística (`LogisticRegression`), treinada no conjunto de treino. Ela aprendeu pesos \([0.0296,\ 0.0015]\) e bias \(-0.0031\), e obteve **47,5%** de acurácia no teste.

### C - Implemente o MLP

| Item | Valor |
|---|---|
| Arquitetura | 2-8-1 (2 entradas, 8 neurônios ocultos, 1 saída) |
| Ativação oculta | \(\tanh\) |
| Ativação de saída | sigmoide |
| Perda | entropia cruzada binária |
| Inicialização | pesos \(\sim \mathcal{N}(0,\ 1/\sqrt{n_{in}})\), bias zero, `rng = np.random.default_rng(42)` |
| Taxa de aprendizado | \(\eta = 0.5\) |
| Épocas | 2000 (gradiente descendente com o conjunto de treino inteiro) |
| Perda de treino | \(0.7930 \rightarrow 0.1043\) |
| Acurácia no teste | **91,5%** |

A Figura 2 mostra a perda caindo ao longo das épocas, indicando que o treinamento convergiu. A Figura 3 mostra as regiões de decisão aprendidas, calculadas sobre uma grade de \(300 \times 300\) pontos, com os pontos de teste sobrepostos.

![Perda de treino por época](figures/fig2_perda_binario.png)
/// caption
**Figura 2** — Perda de treino × época do MLP binário.
///

![Fronteira de decisão](figures/fig3_fronteira.png)
/// caption
**Figura 3** — Fronteira de decisão do MLP sobre os pontos de teste.
///

### D - Análise

**Por que a linha de base fica abaixo de 50%.** A Classe 1 está dividida em dois grupos em cantos opostos do plano, um em torno de \((3, 3)\) e outro em torno de \((-3, -3)\), e a Classe 0 fica no meio, perto da origem. Uma fronteira linear só consegue dividir o plano em dois semiplanos. Para deixar os dois grupos da Classe 1 do mesmo lado, ela teria que cortar a Classe 0 ao meio. Na prática, cada grupo puxa a reta para um lado e um efeito anula o outro: os pesos aprendidos terminaram quase zerados (\([0.0296,\ 0.0015]\)), e a fronteira acabou passando bem no meio do grupo azul. Por isso, mais ou menos metade da Classe 0 e um grupo inteiro da Classe 1 ficam do lado errado, e o modelo não vai além de um chute. Os 47,5%, um pouco abaixo de 50%, são só variação do conjunto de teste.

**Por que o MLP passa de 90%.** Cada neurônio da camada oculta calcula \(\tanh(\mathbf{w}^\top\mathbf{x} + b)\), ou seja, diz de que lado de uma reta o ponto está. A camada de saída junta essas respostas. Com duas retas paralelas, a rede já consegue expressar algo como "é Classe 1 se estiver acima da primeira reta **ou** abaixo da segunda", que é justamente o que uma reta sozinha não dá conta de fazer. Com 8 neurônios, sobra espaço para as bordas se ajustarem melhor aos grupos e até ficarem levemente curvas. No fim, o MLP chegou a 91,5% de acurácia no teste.

**Forma da região da Classe 1.** Na Figura 3, dá para ver que a Classe 0 ocupa uma faixa diagonal em volta da origem, enquanto a Classe 1 aparece em duas regiões separadas, uma de cada lado dessa faixa. Ou seja, a região da Classe 1 é desconexa. Já o lado de uma reta é sempre um único semiplano, conexo, e por isso nenhuma fronteira linear conseguiria reproduzi-la. Os 8,5% de erro que sobram no MLP se concentram perto das bordas da faixa, onde os grupos se sobrepõem por causa do espalhamento (\(\sigma = 1.2\)). Esses pontos caíram no território da outra classe, e não existe fronteira que os acerte sem errar outros no lugar.

### Código

#### Implementação do MLP

``` { .python .copy .select linenums='1' title="docs/exercises/mlp/code/mlp.py" }
--8<-- "docs/exercises/mlp/code/mlp.py"
```

#### Geração e treinamento do Exercise 2

``` { .python .copy .select linenums='1' title="docs/exercises/mlp/code/ex2_binary.py" }
--8<-- "docs/exercises/mlp/code/ex2_binary.py"
```

## Exercise 3

### Abordagem

<!-- 1500 amostras, 4 atributos, 3 classes; mesma classe MLP com saída softmax
     e entropia cruzada categórica. -->

### A - Gere os dados

<!-- Parâmetros do make_classification e o split usado. -->

### B - Adapte a rede

<!-- O que mudou (3 saídas, softmax, entropia cruzada categórica) e o que NÃO mudou
     (forward/backward/update iguais → ponto extra).
     Derivação de ∂L/∂u para softmax + entropia cruzada: o resultado é ŷ − y. -->

### C - Treine e avalie

<!-- Arquitetura, hiperparâmetros e acurácia no teste. -->

![Perda de treino por época - multiclasse](figures/fig4_perda_multiclasse.png)
/// caption
**Figura 4** — Perda de treino × época do modelo multiclasse.
///

![Matriz de confusão](figures/fig5_confusao.png)
/// caption
**Figura 5** — Matriz de confusão no conjunto de teste.
///

### D - Análise

<!-- Par de classes mais confundido e por quê (4 dimensões, 2 grupos por classe = 6 grupos).
     Acurácia da logística (≈66%) e o que a camada oculta acrescentou. -->

### Código

``` { .python .copy .select linenums='1' title="docs/exercises/mlp/code/exercise3_multiclass.py" }
--8<-- "docs/exercises/mlp/code/ex3_multiclass.py"
```

## Exercise 4

### Abordagem

<!-- Mesmos dados, semente, épocas e taxa do Ex. 3; só muda o número de camadas ocultas.
     Explique como tratou as 3 sementes. -->

### A - Treine a rede mais profunda

<!-- Arquitetura (ex.: 4-16-16-3) e acurácia no teste. -->

### B - Compare

![Curvas de perda Ex. 3 vs Ex. 4](figures/fig6_comparacao.png)
/// caption
**Figura 6** — Perda de treino das redes do Exercício 3 e do Exercício 4.
///

| Rede | Camadas ocultas | Nº de parâmetros | Perda final (média) | Acurácia teste (média) |
|---|---|---|---|---|
| Ex. 3 | 1 | | | |
| Ex. 4 | 2 | | | |

### C - Análise

<!-- Por que a rede mais funda não ganhou (ou ganhou)? Considere:
     nº de grupos e complexidade da fronteira; o que a 2ª camada acrescenta vs. o custo de otimizar;
     se perda de treino e acurácia contam a mesma história; que dado recompensaria profundidade. -->

### Código

``` { .python .copy .select linenums='1' title="docs/exercises/mlp/code/exercise4_depth.py" }
--8<-- "docs/exercises/mlp/code/ex4_depth.py"
```

## Results summary

| # | Quantidade | Valor |
|---|---|---|
| 1 | Ex. 1: \(\mathbf{z}^{(1)}\) e \(\mathbf{h}^{(1)}\) | \(\mathbf{z}^{(1)} = [0.2700,\ -0.1800]\), \(\mathbf{h}^{(1)} = [0.2636,\ -0.1781]\) |
| 2 | Ex. 1: \(u^{(2)}\), \(\hat{y}\) e \(L\) | \(0.3852\), \(0.3672\), \(0.4004\) |
| 3 | Ex. 1: parâmetros atualizados | \(\mathbf{W}^{(1)} = [[0.3764,\ -0.1306],[0.1523,\ 0.4191]]\), \(\mathbf{b}^{(1)} = [0.2528,\ -0.2954]\), \(\mathbf{W}^{(2)} = [0.5866,\ -0.3585]\), \(b^{(2)} = 0.5284\) |
| 4 | Ex. 2: acurácia da linha de base linear | 47,5% |
| 5 | Ex. 2: arquitetura e acurácia no teste | 2-8-1 (tanh / sigmoide), 91,5% |
| 6 | Ex. 3: arquitetura e acurácia no teste | |
| 7 | Ex. 3: par de classes mais confundido | |
| 8 | Ex. 4: arquitetura e acurácia no teste | |
| 9 | Ex. 4: diferença de acurácia em relação ao Ex. 3 (média de 3 sementes) | |

## Discussão

<!-- Um parágrafo ligando os exercícios: o Ex. 1 mostra o mecanismo (um passo);
     o Ex. 2 mostra por que a camada oculta importa (fronteira não linear);
     o Ex. 3 mostra que a mesma rede serve para multiclasse;
     o Ex. 4 mostra que mais profundidade nem sempre ajuda. -->

## Conclusão

<!-- 2–3 frases com o principal aprendizado. -->
