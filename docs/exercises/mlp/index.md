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

Os dados foram gerados com `make_classification`, usando `n_samples=1500`, `n_features=4`, `n_informative=4`, `n_redundant=0`, `n_repeated=0`, `n_classes=3`, `n_clusters_per_class=2`, `class_sep=1.0` e `random_state=42`. São 3 classes com 500 amostras cada, e cada classe é formada por 2 grupos, totalizando 6 grupos em um espaço de 4 dimensões. A divisão treino/teste usou `train_test_split` com `test_size=0.2`, `random_state=42` e `stratify=y`, resultando em 1200 amostras de treino e 300 de teste (100 por classe).

### B - Adapte a rede

Em relação ao Exercício 2, só a saída da rede mudou:

| | Exercício 2 | Exercício 3 |
|---|---|---|
| Unidades de saída | 1 | 3 (uma por classe) |
| Ativação de saída | sigmoide | softmax |
| Perda | entropia cruzada binária | entropia cruzada categórica |
| Alvo | \(y \in \{0, 1\}\) | \(\mathbf{y}\) em one-hot |

A softmax transforma as pré-ativações de saída \(\mathbf{u} = (u_1, u_2, u_3)\) em probabilidades que somam 1:

\[
\hat{y}_k = \frac{e^{u_k}}{\sum_j e^{u_j}}
\]

e a entropia cruzada categórica, para uma amostra com alvo one-hot \(\mathbf{y}\), é:

\[
L = -\sum_k y_k \log \hat{y}_k
\]

**Derivação de \(\partial L / \partial u_i\).** Tomando o log da softmax:

\[
\log \hat{y}_k = u_k - \log \sum_j e^{u_j}
\]

Substituindo na perda:

\[
L = -\sum_k y_k u_k + \Big(\sum_k y_k\Big) \log \sum_j e^{u_j}
\]

Como \(\mathbf{y}\) é one-hot, \(\sum_k y_k = 1\), então:

\[
L = -\sum_k y_k u_k + \log \sum_j e^{u_j}
\]

Derivando em relação a \(u_i\): o primeiro termo dá \(-y_i\), e o segundo dá \(\dfrac{e^{u_i}}{\sum_j e^{u_j}} = \hat{y}_i\). Portanto:

\[
\boxed{\frac{\partial L}{\partial u_i} = \hat{y}_i - y_i}
\]

Esse é o mesmo gradiente obtido no Exercício 2 com sigmoide e entropia cruzada binária (\(\partial L/\partial u = \hat{y} - y\)). Por isso, o início da retropropagação é idêntico nos dois casos.

**Reaproveitamento do código.** O arquivo `mlp.py` não foi alterado entre os Exercícios 2 e 3: os métodos `forward`, `backward` e `update` são exatamente os mesmos. A única diferença está nos argumentos passados à classe:

```python
model = MLP(layer_sizes=[4, 16, 3], rng=rng, learning_rate=0.5, epochs=2000, output="softmax")
```

Com `output="softmax"`, a classe aplica softmax na última camada, usa a entropia cruzada categórica e converte os rótulos para one-hot. O `backward` começa com \(\boldsymbol{\delta} = (\hat{\mathbf{y}} - \mathbf{y})/N\) nos dois exercícios, conforme a derivação acima.

### C - Treine e avalie

| Item | Valor |
|---|---|
| Arquitetura | 4-16-3 (4 entradas, 16 neurônios ocultos, 3 saídas) |
| Ativação oculta | \(\tanh\) |
| Ativação de saída | softmax |
| Perda | entropia cruzada categórica |
| Inicialização | pesos \(\sim \mathcal{N}(0,\ 1/\sqrt{n_{in}})\), bias zero, `rng = np.random.default_rng(42)` |
| Taxa de aprendizado | \(\eta = 0.5\) |
| Épocas | 2000 (gradiente descendente com o conjunto de treino inteiro) |
| Perda de treino | \(1.1166 \rightarrow 0.2597\) |
| Acurácia no teste | **85,67%** |

A acurácia ficou dentro da faixa esperada (83–86%). A Figura 4 mostra a perda de treino caindo ao longo das épocas, e a Figura 5 mostra a matriz de confusão no conjunto de teste: a diagonal (90, 77 e 90) corresponde aos acertos de cada classe, e a Classe 1 é a que tem mais erros.


![Perda de treino por época - multiclasse](figures/fig4_perda_multiclasse.png)
/// caption
**Figura 4** — Perda de treino × época do modelo multiclasse.
///

![Matriz de confusão](figures/fig5_confusao.png)
/// caption
**Figura 5** — Matriz de confusão no conjunto de teste.
///

### D - Análise

**Par mais confundido.** Pela Figura 5, o par que o modelo mais confunde é Classe 0 ↔ Classe 1 (6 + 12 = 18 erros), seguido de perto por Classe 1 ↔ Classe 2 (11 + 5 = 16). Já Classe 0 ↔ Classe 2 é o par menos confundido (4 + 5 = 9). A Classe 1 é a que tem menos acertos (77 de 100), já que é trocada tanto pela Classe 0 quanto pela Classe 2.

Isso tem a ver com a estrutura dos dados. O `make_classification` coloca os 6 grupos (3 classes × 2 grupos) perto dos vértices de um hipercubo em 4 dimensões, com coordenadas próximas de \(\pm 1\). Estimando os centros dos grupos com K-means, dá para ver que um dos grupos da Classe 1 difere de um grupo da Classe 0 em apenas uma coordenada, e o outro grupo da Classe 1 difere de um grupo da Classe 2 também em uma só. Ou seja, são vértices vizinhos, e os pontos acabam se sobrepondo na fronteira entre eles. Os grupos da Classe 0 e da Classe 2, por outro lado, diferem em pelo menos duas coordenadas e ficam mais afastados. É por isso que a Classe 1 fica "no meio" e concentra a maior parte dos erros.

**Linha de base.** A regressão logística chegou a **66,33%** no teste, contra **85,67%** do MLP, um ganho de cerca de 19 pontos percentuais.

**O que a camada oculta acrescentou.** Assim como no Exercício 2, cada classe é formada por dois grupos separados. Um modelo linear atribui a cada classe uma única região delimitada por hiperplanos, então não consegue cercar dois grupos distantes da mesma classe sem acabar incluindo grupos de outras classes no caminho. A camada oculta resolve isso combinando várias fronteiras lineares (cada neurônio define um hiperplano), o que permite isolar cada grupo separadamente. Os erros que sobram ficam justamente onde grupos vizinhos de classes diferentes se sobrepõem.

### Código

``` { .python .copy .select linenums='1' title="docs/exercises/mlp/code/ex3_multiclass.py" }
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
| 6 | Ex. 3: arquitetura e acurácia no teste | 4-16-3 (tanh / softmax), 85,67% |
| 7 | Ex. 3: par de classes mais confundido | Classe 0 ↔ Classe 1 (18 erros) |
| 8 | Ex. 4: arquitetura e acurácia no teste | |
| 9 | Ex. 4: diferença de acurácia em relação ao Ex. 3 (média de 3 sementes) | |

## Discussão

<!-- Um parágrafo ligando os exercícios: o Ex. 1 mostra o mecanismo (um passo);
     o Ex. 2 mostra por que a camada oculta importa (fronteira não linear);
     o Ex. 3 mostra que a mesma rede serve para multiclasse;
     o Ex. 4 mostra que mais profundidade nem sempre ajuda. -->

## Conclusão

<!-- 2–3 frases com o principal aprendizado. -->
