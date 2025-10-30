# Sprint Challenge 4: Previsão de Gravidade de Acidentes com LSTMs (PRF)

**Autores:** Octavio, Guilherme

Este projeto é uma solução de Deep Learning desenvolvida para o Sprint Challenge 4, com o objetivo de prever a gravidade de acidentes de trânsito nas rodovias federais brasileiras. Utilizando Redes Neurais Recorrentes (LSTMs) e a base de dados pública da Polícia Rodoviária Federal (PRF), o modelo visa identificar padrões temporais que levam a acidentes mais severos.

---

## 1. O Problema: Target Escolhido

O objetivo deste projeto foi construir um modelo capaz de prever a **gravidade de um acidente**. Para isso, o problema foi modelado como uma **classificação binária**.

O *target* (`y`) foi criado a partir das colunas de fatalidade e ferimentos do dataset:

* **Classe 0 (Minoritária):** Acidentes Graves ou Fatais (onde `mortos > 0` ou `feridos_graves > 0`).
* **Classe 1 (Majoritária):** Acidentes Leves ou Sem Vítimas (onde `feridos_leves > 0` ou `ilesos > 0`, e sem mortes ou feridos graves).

A justificativa para esta escolha é o alto impacto de se conseguir antecipar os fatores que levam a desfechos trágicos (Classe 0). Um modelo preciso pode auxiliar na alocação de recursos, fiscalização preventiva e até na precificação de seguros de forma mais justa, focando nos cenários de maior risco.

## 2. Metodologia e Pré-processamento

O tratamento dos dados foi a etapa mais crítica do projeto, focada em dois desafios principais: a formatação dos dados para a LSTM e o combate ao desequilíbrio de classes.

### 2.1. Desafio 1: O Desequilíbrio Extremo

O principal desafio encontrado foi o severo desequilíbrio de classes. A Classe 0 (Grave/Fatal) representava uma fração minúscula do total de registros.

Durante o desenvolvimento, foi identificado que uma simples divisão temporal (ex: 70% para treino, 30% para teste) escondia todos os casos raros do conjunto de treinamento, levando o modelo a atingir um `recall` de 0.00 para a Classe 0.

### 2.2. A Solução (em 3 Passos)

1.  **Divisão Estratificada (`stratify=y`)**: A divisão sequencial foi substituída pelo `train_test_split` do Scikit-learn com o parâmetro `stratify=y`. Isso garantiu que a proporção de casos raros fosse mantida nos conjuntos de treino, validação e teste.
2.  **Balanceamento com `RandomUnderSampler`**: Após tentativas com `class_weight` e `SMOTE` (Oversampling) não apresentarem os resultados desejados, a técnica de **Undersampling** foi aplicada. O `RandomUnderSampler` do `imbalanced-learn` foi usado para reduzir o conjunto de *treino* a um balanço perfeito de 50/50 entre as classes, forçando o modelo a aprender os padrões da Classe 0.
3.  **Otimização do Limite de Decisão**: O modelo treinado com dados balanceados alcançou um excelente `ROC-AUC`, mas o `threshold` (limite) padrão de 0.5 ainda era impreciso. Foi implementada uma rotina de otimização para encontrar o limite que maximizava o `f1-score` da Classe 0 (Minoritária).

## 3. Arquitetura do Modelo

Foi utilizada uma arquitetura LSTM simples, focada em classificação binária:

1.  `Input: LSTM (64 unidades)`
2.  `Dropout (0.3)`
3.  `Dense (32 unidades, ativação 'relu')`
4.  `Output: Dense (1 unidade, ativação 'sigmoid')`

O modelo foi compilado com o otimizador `adam`, `loss='binary_crossentropy'` e monitorando a métrica `AUC`.

## 4. Resultados e Avaliação

O modelo foi avaliado em seu `ROC-AUC` (que mede a capacidade de discriminação) e pelo `classification_report`, que detalha o desempenho em cada classe.

O modelo final, treinado com Undersampling e avaliado com o limite de decisão otimizado, alcançou um **ROC-AUC de 0.9516**.

### Métricas Finais (Limite Otimizado)

O relatório abaixo demonstra o sucesso do modelo em encontrar um equilíbrio entre precisão e recall para a classe minoritária.

| Classe | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Grave/Fatal)** | **0.67** | **0.50** | **0.57** | 4 |
| 1 (Leve/Ileso) | 0.97 | 0.98 | 0.98 | 62 |
| **Accuracy** | | | **0.95** | 66 |
| **Macro Avg** | **0.82** | **0.74** | **0.77** | 66 |
| **Weighted Avg** | **0.95** | **0.95** | **0.95** | 66 |

## 5. Conteúdo do Repositório

* `Seu_Notebook.ipynb`: O notebook Jupyter/Colab com todo o processo de ETL, pré-processamento, treinamento e avaliação.
* `best_model.keras`: O arquivo do modelo final treinado, salvo pelo `ModelCheckpoint`.
* `scaler.joblib`: O objeto `StandardScaler` treinado.
* `labelencoder_uf.joblib`: O objeto `LabelEncoder` treinado (se aplicável).
* `Relatorio.pdf`: (Opcional) O relatório de explicação, se feito à parte.
* `README.md`: Este arquivo.

## 6. Como Executar o Projeto

1.  **Clonar o Repositório:**
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DO_REPOSITORIO]
    ```

2.  **Instalar Dependências:**
    É recomendado o uso de um ambiente virtual (`venv` ou `conda`).
    ```bash
    pip install pandas numpy matplotlib tensorflow scikit-learn imbalanced-learn joblib
    ```

3.  **Dataset:**
    Baixe a base de dados da PRF (link no desafio) e coloque-a no diretório `data/` (ou ajuste o caminho no notebook).

4.  **Execução:**
    Abra o `Seu_Notebook.ipynb` em um ambiente como Jupyter ou Google Colab e execute as células na ordem.

5.  **Carregando o Modelo (Opcional):**
    Para carregar o modelo pré-treinado e pular o treinamento:
    ```python
    import tensorflow as tf
    best_model = tf.keras.models.load_model('best_model.keras')
    best_model.summary()
    ```
## Link para o Notebook (Google Colab)

Todo o processo de desenvolvimento, pré-processamento e treinamento pode ser visualizado e executado diretamente no Google Colab através do link abaixo:

**[Abrir o Notebook no Google Colab](https://colab.research.google.com/drive/10u83CeIL97tmQRlt0QrbRWrLbs6PlKF9)**
