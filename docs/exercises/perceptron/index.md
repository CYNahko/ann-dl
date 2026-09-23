---
exercise: perceptron
ai_use: "Utilizado para esclarescer o que o exercício pedia e auxílio na formatação da página."
---

# 2. Perceptron

!!! abstract "Enunciado"

    [Exercises → Perceptron](https://insper.github.io/ann-dl/){:target='_blank'}


## Exercise 1

### A - Gere os dados

### Abordagem

Durante a execução do exercício 1, foram geradas duas classes: Classe 0 e Classe 1, com 1000 amostras cada a partir de distribuições normais multivariadas, utlizando as médias e a matriz de covariância definidas no enunciado. As amostras foram reunidas em uma matriz de atributos e classificadas por uma implementação própria do perceptron, com taxa de aprendizagem de \(0.01\) e limite de 100 épocas. Durante o treinamento, foram registradas a acurácia por época e as atualizações dos parâmetros. Por fim, foram visualizadas a distribuição dos dados, a fronteira de decisão e a evolução da acurácia.

### Código

#### Implementação do perceptron
``` { .python .copy .select linenums='1' title="docs/exercises/perceptron/code/perceptron_class.py" }
--8<-- "docs/exercises/perceptron/code/perceptron_class.py"
```

#### Geração e Treinamento do Exercise 1

``` { .python .copy .select linenums='1' title="docs/exercises/perceptron/code/exercise1_separatable_data.py" }
--8<-- "docs/exercises/perceptron/code/exercise1_separatable_data.py"
```

### Figuras

![Gráfico de Dispersão dos 2000 pontos](figures/dispersion_grafic.png)
/// caption
**Figura 1** — Gráfico de Dispersão dos 2000 pontos.
///

![Fronteira de decisão do Perceptron](figures/decision_frontier_perceptron.png)
/// caption
**Figura 2** — Fronteira de decisão do Perceptron
///

![Acurácia por Época](figures/accuracy_per_epoch.png)
/// caption
**Figura 3** — Acurácia por Época
///

### Análise

1. Dados separáveis convergem rápido, pois existe uma reta capaz de separar corretamente todas as amostras das duas classes. A cada erro, o perceptron ajusta seus pesos e aproxima a fronteira dessa separação. Como há pouca ou nenhuma sobreposição entre as classes, as correções não entram constantemente em conflito. Assim que uma fronteira válida é encontrada, nenhuma nova atualização acontece e o treinamento termina. No caso do exercício, as médias \([1.5,1.5]\) e \([5,5]\) estão distantes, enquanto a variância é pequena. Por isso, os grupos ficam bem separados e o perceptron alcançou 100% de acurácia em apenas 26 épocas.

2. Os dois treinamentos alcançaram 100% de acurácia. Com \(\eta=0.01\), o perceptron convergiu em 34 épocas e produziu a direção normalizada \([0.8543,\ 0.5197]\). Com \(\eta=1.0\), convergiu em 37 épocas e produziu a direção \([0.8691,\ 0.4946]\). Portanto, aumentar a taxa não tornou a convergência necessariamente mais rápida. Como cada atualização soma \(\eta\mathbf{x}\) aos pesos, a taxa controla o tamanho dos passos em relação aos pesos iniciais, fazendo os treinamentos percorrerem trajetórias diferentes e encontrarem fronteiras ligeiramente distintas.

3. Caso os pesos e o bias começassem em zero, mudar a taxa de aprendizagem apenas aumentaria ou diminuiria proporcionalmente todos os valores aprendidos. Essa mudança de escala não alteraria as previsões, pois os pontos continuariam do mesmo lado da fronteira de decisão. Portanto, as atualizações aconteceriam na mesma ordem e os treinamentos levariam a mesma quantidade de épocas. É por isso que o exercício exige pesos iniciais aleatórios diferentes de zero: assim, a taxa de aprendizagem realmente pode influenciar o caminho do treinamento e a fronteira encontrada.

## Exercise 2

### Abordagem

Foram geradas duas classes parcialmente sobrepostas, com 1.000 amostras cada, utilizando as médias e a matriz de covariância definidas no enunciado. A mesma implementação do perceptron do Exercício 1 foi treinada por 100 épocas com taxa de aprendizagem de \(0.01\). Durante o treinamento, o algoritmo pocket guardou uma cópia dos pesos sempre que uma atualização produziu uma nova melhor acurácia. Ao final, foram comparados os pesos finais com os melhores pesos armazenados no pocket.

### Código

#### Geração e Treinamento do Exercise 2

``` { .python .copy .select linenums='1' title="docs/exercises/perceptron/code/exercise2_overlapping_data.py" }
--8<-- "docs/exercises/perceptron/code/exercise2_overlapping_data.py"
```

### Figuras

![Gráfico de Dispersão dos 2000 pontos](figures/dispersion_grafic.png)
/// caption
**Figura 1** — Gráfico de Dispersão dos 2000 pontos.
///

![Fronteira de decisão do Perceptron](figures/decision_frontier_perceptron.png)
/// caption
**Figura 2** — Fronteira de decisão do Perceptron
///

![Acurácia por Época](figures/accuracy_per_epoch.png)
/// caption
**Figura 3** — Acurácia por Época
///

### Análise



## Results summary

| # | Quantidade | Valor |
|---:|------------|-------|
| 1 | Exercício 1 — $\mathbf{w}$ e $b$ finais | $\mathbf{w}=[0.0518,\ 0.0315]$; $b=$ ... |
| 2 | Exercício 1 — épocas até convergir | 34 |
| 3 | Exercício 1 — acurácia final | 100% |
| 4 | Exercício 1 — épocas e acurácia com $\eta=1.0$ | 37 épocas; 100% |
| 5 | Exercício 2 — $\mathbf{w}$ e $b$ finais | A preencher |
| 6 | Exercício 2 — acurácia dos pesos finais | A preencher |
| 7 | Exercício 2 — acurácia dos pesos do *pocket* | A preencher |
| 8 | Exercício 2 — melhor época do *pocket* | A preencher |

## Discussão

## Conclusão