# Relatório do Sprint Challenge 4: Previsão de Gravidade de Acidentes com LSTMs

**Autores:** Octavio, Guilherme

**Projeto:** Previsão da Gravidade de Acidentes de Trânsito com Redes Neurais Recorrentes (LSTM)
**Base de Dados:** Polícia Rodoviária Federal (PRF)

---

### 1. Target Escolhido e Justificativa

O objetivo central deste projeto foi desenvolver um modelo de *Deep Learning* capaz de prever a **severidade de um acidente** de trânsito. Para isso, o problema foi estruturado como uma **classificação binária**.

O *target* (variável-alvo) foi criado a partir das colunas de fatalidade e ferimentos do dataset:

* **Classe 0 (Minoritária):** Acidentes Graves ou Fatais (definidos como qualquer registro onde `mortos > 0` ou `feridos_graves > 0`).
* **Classe 1 (Majoritária):** Acidentes Leves ou Sem Vítimas (registros sem mortes ou feridos graves).

**Justificativa:** A escolha foi motivada pelo alto impacto social e estratégico de se prever os fatores que levam a acidentes com desfechos trágicos. A Classe 0, embora rara, é a mais custosa (humana e financeiramente). Um modelo capaz de identificar os padrões de risco para esta classe pode auxiliar diretamente na alocação de recursos de emergência, planejamento de fiscalização e em estratégias de prevenção, focando nos cenários de maior perigo.

### 2. Tratamento e Pré-processamento dos Dados

Esta foi a etapa mais desafiadora e crucial do projeto. O *dataset* apresentava um **desequilíbrio de classes extremo**, com os acidentes da Classe 0 (Graves/Fatais) sendo eventos muito raros em comparação com a Classe 1.

Nossa jornada de pré-processamento evoluiu da seguinte forma:

1.  **A Descoberta do Problema Raiz:** Nossas tentativas iniciais (usando `class_weight`, `SMOTE`) falharam, resultando em um `recall` de 0.00 para a Classe 0. A investigação revelou que a divisão de dados original, baseada em tempo (ex: `X[:n_train]`), acidentalmente colocou **todas** as amostras raras da Classe 0 no conjunto de teste, fazendo com que o modelo fosse treinado sem nunca ter visto um único exemplo de acidente grave.

2.  **Solução de Divisão: `train_test_split(stratify=y)`**
    A correção fundamental foi abandonar a divisão temporal e adotar uma divisão aleatória **estratificada**. Ao usar `stratify=y`, garantimos que a proporção minúscula de casos da Classe 0 fosse distribuída corretamente entre os conjuntos de treino, validação e teste.

3.  **Solução de Balanceamento: `RandomUnderSampler`**
    Com os dados agora divididos corretamente, o conjunto de treino (ainda que estratificado) permanecia desbalanceado. Aplicamos a técnica de **Undersampling** (subamostragem) *apenas no conjunto de treino*. Isso criou um novo conjunto de treino 50/50, forçando o modelo a dar igual importância aos padrões de ambas as classes.

4.  **Solução de Avaliação: Otimização do Limite de Decisão**
    O modelo treinado com *Undersampling* alcançou um excelente `ROC-AUC` (0.95), provando que ele *sabia* diferenciar as classes. No entanto, o limite de decisão (`threshold`) padrão de 0.5 ainda classificava mal a Classe 0. Implementamos uma rotina de otimização que varreu 100 limites diferentes (de 0.01 a 0.99) para encontrar aquele que maximizava o `F1-Score` da Classe 0, destravando o verdadeiro desempenho do modelo.

### 3. Arquitetura do Modelo LSTM

Para a classificação das sequências, foi implementada uma Rede Neural Recorrente (LSTM) simples, mas eficaz:

1.  **Camada de Entrada:** `LSTM` com 64 unidades, recebendo o `input_shape` dos dados (`timesteps`, `n_features`) e com `return_sequences=False`, pois estávamos interessados apenas na classificação da sequência inteira.
2.  **Regularização:** `Dropout` de 0.3 (30%) para prevenir overfitting.
3.  **Camada Oculta:** `Dense` (Totalmente Conectada) com 32 unidades e ativação `relu`.
4.  **Camada de Saída:** `Dense` com 1 unidade e ativação `sigmoid`, ideal para classificação binária.

**Compilação:** O modelo foi compilado com o otimizador `Adam`, função de perda `binary_crossentropy` e, crucialmente, monitorando a métrica `AUC` (Area Under the Curve), que é a mais indicada para avaliar a performance de discriminação em cenários desbalanceados.

### 4. Métricas e Avaliação de Resultados

As métricas de avaliação foram escolhidas para refletir um cenário desbalanceado. A Acurácia (`accuracy`) sozinha seria uma métrica enganosa.

* **Métrica de Treinamento:** `ROC-AUC` foi usada para avaliar a capacidade geral do modelo de distinguir entre as duas classes.
* **Métricas de Desempenho:** O `classification_report` (Precision, Recall, F1-Score), com foco especial no **Recall** e **F1-Score** da **Classe 0**.

**Resultados Finais (com Undersampling e Limite Otimizado):**

O modelo final alcançou um **ROC-AUC de 0.9516**, indicando um poder de discriminação excelente.

O relatório de classificação, usando o limite otimizado, demonstrou o sucesso do modelo em finalmente "enxergar" a classe minoritária:

| Classe | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Grave/Fatal)** | **0.67** | **0.50** | **0.57** | 4 |
| 1 (Leve/Ileso) | 0.97 | 0.98 | 0.98 | 62 |
| **Accuracy** | | | **0.95** | 66 |
| **Macro Avg** | **0.82** | **0.74** | **0.77** | 66 |

**Conclusão:** O modelo final é um sucesso. Ele foi capaz de **detectar 50% de todos os acidentes graves/fatais** (Recall de 0.50) no conjunto de teste, algo que era 0% no início do projeto. Além disso, quando ele prevê um acidente como grave (Precision de 0.67), ele está correto em 2 de cada 3 vezes, provando ser uma ferramenta valiosa e funcional para a análise de risco.
