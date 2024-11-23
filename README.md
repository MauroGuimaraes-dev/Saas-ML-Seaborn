# Máquina Preditiva com Datasets Seaborn

Uma aplicação web interativa para análise de dados e machine learning, construída com Streamlit e scikit-learn, seguindo o padrão MVC (Model-View-Controller).

## Estrutura do Projeto

O projeto segue uma arquitetura MVC clara e organizada:

```
ChatPredML/
├── models/                 # Camada de Modelo
│   ├── __init__.py
│   ├── data_model.py      # Gerenciamento de dados
│   └── ml_model.py        # Modelos de machine learning
├── views/                  # Camada de Visualização
│   ├── __init__.py
│   └── data_view.py       # Interface do usuário
├── controllers/           # Camada de Controle
│   ├── __init__.py
│   └── ml_controller.py   # Lógica de controle principal
├── app.py                # Arquivo principal
└── requirements.txt      # Dependências do projeto
```

## Componentes Principais

### 1. Models (Modelos)

#### data_model.py
- Gerencia os datasets disponíveis do Seaborn
- Prepara dados para treinamento
- Identifica colunas numéricas
- Principais métodos:
  - `get_available_datasets()`: Retorna datasets disponíveis
  - `prepare_data()`: Prepara dados para ML
  - `get_numeric_columns()`: Retorna colunas numéricas

#### ml_model.py
- Gerencia algoritmos de machine learning
- Treina e avalia modelos
- Principais métodos:
  - `get_available_algorithms()`: Retorna algoritmos disponíveis
  - `train_and_evaluate()`: Treina e avalia modelos

### 2. Views (Visualizações)

#### data_view.py
- Gerencia a interface do usuário
- Cria visualizações de dados
- Exibe resultados
- Principais métodos:
  - `show_dataset_info()`: Mostra informações do dataset
  - `show_visualizations()`: Cria gráficos
  - `show_model_result()`: Exibe resultados do modelo
  - `show_error_message()`: Exibe mensagens de erro

### 3. Controllers (Controladores)

#### ml_controller.py
- Coordena interação entre modelos e views
- Gerencia fluxo da aplicação
- Principais métodos:
  - `initialize_page()`: Configura página inicial
  - `get_user_selections()`: Obtém seleções do usuário
  - `process_data()`: Processa dados selecionados
  - `train_model()`: Treina e avalia modelo

### 4. Arquivo Principal (app.py)
- Ponto de entrada da aplicação
- Inicializa o controlador
- Executa o fluxo principal

## Funcionalidades

1. **Seleção de Dataset**
   - Datasets disponíveis: Tips, Iris, Diamonds, Penguins
   - Visualização de amostra dos dados
   - Estatísticas descritivas

2. **Visualizações**
   - Histograma da primeira variável numérica
   - Gráfico de dispersão das duas primeiras variáveis

3. **Machine Learning**
   - Algoritmos disponíveis:
     - Regressão Linear
     - Árvore de Decisão
     - Random Forest
   - Treinamento automático
   - Avaliação com R² Score

## Dependências

- streamlit>=1.24.0
- seaborn>=0.12.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- numpy>=1.24.0
- matplotlib>=3.7.0

## Como Executar

1. Clone o repositório
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute a aplicação:
   ```bash
   streamlit run app.py
   ```

## Arquitetura MVC

### Model (Modelo)
- Responsável pelo acesso e manipulação dos dados
- Implementa a lógica de negócio
- Gerencia os algoritmos de ML

### View (Visualização)
- Responsável pela interface do usuário
- Cria visualizações dos dados
- Exibe resultados e mensagens

### Controller (Controlador)
- Coordena a interação entre Model e View
- Gerencia o fluxo da aplicação
- Processa as entradas do usuário

## Algoritmos e Hiperparâmetros

### 1. Regressão Linear
Modelo linear simples para prever valores contínuos.

**Hiperparâmetros:**
- `fit_intercept` (Padrão: True)
  - Descrição: Permite incluir ou não o termo de interceptação
  - Valores: True/False
  - Uso: Se True, o modelo calculará o intercepto (termo constante). Se False, o modelo assumirá que os dados estão centralizados.

- `normalize` (Padrão: False)
  - Descrição: Opção para normalizar os dados (deprecated)
  - Valores: True/False
  - Nota: Recomenda-se usar StandardScaler antes do modelo

### 2. Árvore de Decisão
Modelo baseado em árvore para prever valores contínuos.

**Hiperparâmetros:**
- `max_depth` (Padrão: None)
  - Descrição: Profundidade máxima da árvore
  - Valores: 1 a 20 ou None
  - Uso: Se None, a árvore crescerá até as folhas serem puras ou terem min_samples_split amostras

- `min_samples_split` (Padrão: 2)
  - Descrição: Mínimo de amostras para dividir um nó
  - Valores: 2 a 20
  - Uso: Número mínimo de amostras necessário para dividir um nó interno

- `min_samples_leaf` (Padrão: 1)
  - Descrição: Mínimo de amostras em cada folha
  - Valores: 1 a 20
  - Uso: Número mínimo de amostras necessário para ser um nó folha

- `random_state` (Padrão: 42)
  - Descrição: Semente aleatória
  - Valores: Qualquer inteiro
  - Uso: Controla a aleatoriedade do estimador para reprodutibilidade

### 3. Random Forest
Ensemble de árvores de decisão para prever valores contínuos.

**Hiperparâmetros:**
- `n_estimators` (Padrão: 100)
  - Descrição: Número de árvores na floresta
  - Valores: 10 a 200
  - Uso: Quantidade de árvores de decisão no ensemble

- `max_depth` (Padrão: None)
  - Descrição: Profundidade máxima das árvores
  - Valores: 1 a 20 ou None
  - Uso: Se None, as árvores crescerão até as folhas serem puras ou terem min_samples_split amostras

- `min_samples_split` (Padrão: 2)
  - Descrição: Mínimo de amostras para dividir um nó
  - Valores: 2 a 20
  - Uso: Número mínimo de amostras necessário para dividir um nó interno

- `min_samples_leaf` (Padrão: 1)
  - Descrição: Mínimo de amostras em cada folha
  - Valores: 1 a 20
  - Uso: Número mínimo de amostras necessário para ser um nó folha

- `random_state` (Padrão: 42)
  - Descrição: Semente aleatória
  - Valores: Qualquer inteiro
  - Uso: Controla a aleatoriedade do estimador para reprodutibilidade

## Métricas de Avaliação

- **R² Score (Coeficiente de Determinação)**
  - Varia de -∞ a 1
  - R² = 1: ajuste perfeito
  - R² = 0: modelo equivalente à média
  - R² < 0: modelo pior que a média

## Boas Práticas Implementadas

1. **Separação de Responsabilidades**
   - Cada componente tem uma função específica
   - Código organizado e manutenível

2. **Documentação**
   - Comentários detalhados em cada arquivo
   - Docstrings explicativas
   - README completo

3. **Código Limpo**
   - Nomes descritivos
   - Funções com responsabilidade única
   - Estrutura clara e organizada

4. **Modularização**
   - Componentes independentes
   - Fácil manutenção e extensão
   - Reutilização de código
