# Importando os algoritmos de machine learning do scikit-learn
from sklearn.linear_model import LinearRegression      # Para regressão linear
from sklearn.tree import DecisionTreeRegressor        # Para árvore de decisão
from sklearn.ensemble import RandomForestRegressor    # Para random forest
import streamlit as st                               # Para interface do usuário

class MLModel:
    """
    Classe responsável por gerenciar os modelos de machine learning.
    Fornece métodos para obter algoritmos disponíveis e treinar/avaliar modelos.
    """
    
    @staticmethod
    def get_model_params(model_name):
        """
        Retorna os hiperparâmetros disponíveis para cada modelo.
        Args:
            model_name (str): Nome do modelo selecionado
        Returns:
            dict: Dicionário com os hiperparâmetros e seus valores
        """
        if model_name == "Regressão Linear":
            # Regressão Linear tem poucos hiperparâmetros ajustáveis
            # Para normalização, use StandardScaler antes de aplicar o modelo
            return {
                "fit_intercept": st.checkbox(
                    "Incluir Intercepto",
                    value=True,
                    help="Se True, o modelo calculará o intercepto (termo constante). "
                         "Se False, o modelo assumirá que os dados estão centralizados."
                )
            }
        
        elif model_name == "Árvore de Decisão":
            # Hiperparâmetros principais da Árvore de Decisão
            return {
                "max_depth": st.slider(
                    "Profundidade Máxima",
                    min_value=1,
                    max_value=20,
                    value=None,
                    help="Profundidade máxima da árvore. Se None, a árvore crescerá até as folhas serem puras "
                         "ou terem min_samples_split amostras."
                ),
                "min_samples_split": st.slider(
                    "Mínimo de Amostras para Divisão",
                    min_value=2,
                    max_value=20,
                    value=2,
                    help="Número mínimo de amostras necessário para dividir um nó interno."
                ),
                "min_samples_leaf": st.slider(
                    "Mínimo de Amostras por Folha",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="Número mínimo de amostras necessário para ser um nó folha."
                ),
                "random_state": st.number_input(
                    "Semente Aleatória",
                    value=42,
                    help="Controla a aleatoriedade do estimador."
                )
            }
        
        elif model_name == "Random Forest":
            # Hiperparâmetros principais do Random Forest
            return {
                "n_estimators": st.slider(
                    "Número de Árvores",
                    min_value=10,
                    max_value=200,
                    value=100,
                    help="Número de árvores na floresta."
                ),
                "max_depth": st.slider(
                    "Profundidade Máxima",
                    min_value=1,
                    max_value=20,
                    value=None,
                    help="Profundidade máxima das árvores. Se None, as árvores crescerão até as folhas serem puras "
                         "ou terem min_samples_split amostras."
                ),
                "min_samples_split": st.slider(
                    "Mínimo de Amostras para Divisão",
                    min_value=2,
                    max_value=20,
                    value=2,
                    help="Número mínimo de amostras necessário para dividir um nó interno."
                ),
                "min_samples_leaf": st.slider(
                    "Mínimo de Amostras por Folha",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="Número mínimo de amostras necessário para ser um nó folha."
                ),
                "random_state": st.number_input(
                    "Semente Aleatória",
                    value=42,
                    help="Controla a aleatoriedade do estimador."
                )
            }
        return {}
    
    @staticmethod
    def get_available_algorithms():
        """
        Retorna um dicionário com os algoritmos de machine learning disponíveis.
        Cada algoritmo é instanciado com seus parâmetros padrão.
        Returns:
            dict: Dicionário com nome do algoritmo e sua instância
        """
        # Obtém o nome do algoritmo selecionado
        algorithm_name = st.selectbox(
            "Selecione o Algoritmo:",
            ["Regressão Linear", "Árvore de Decisão", "Random Forest"]
        )
        
        # Obtém os parâmetros do modelo selecionado
        params = MLModel.get_model_params(algorithm_name)
        
        # Cria o modelo com os parâmetros selecionados
        if algorithm_name == "Regressão Linear":
            model = LinearRegression(**params)
        elif algorithm_name == "Árvore de Decisão":
            model = DecisionTreeRegressor(**params)
        elif algorithm_name == "Random Forest":
            model = RandomForestRegressor(**params)
            
        # Exibe informações sobre o modelo
        st.markdown("""
        <style>
        .model-info {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .model-title {
            color: #1E395B;
            font-weight: bold;
            font-size: 1.2em;
        }
        .param-name {
            color: #1E395B;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="model-info">
            <div class="model-title">Informações do Modelo: {algorithm_name}</div>
            <p>Parâmetros configurados:</p>
            {''.join(f'<p><span class="param-name">{k}:</span> {v}</p>' for k, v in params.items())}
        </div>
        """, unsafe_allow_html=True)
        
        return {algorithm_name: model}
    
    @staticmethod
    def train_and_evaluate(algorithm, X_train, X_test, y_train, y_test):
        """
        Treina um algoritmo com os dados de treino e avalia com os dados de teste.
        Args:
            algorithm: Instância do algoritmo de ML a ser treinado
            X_train: Features de treino
            X_test: Features de teste
            y_train: Target de treino
            y_test: Target de teste
        Returns:
            tuple: (Score R², nome do modelo, dicionário de parâmetros)
        """
        # Treina o modelo com os dados de treino
        algorithm.fit(X_train, y_train)
        
        # Obtém o score R² nos dados de teste
        score = algorithm.score(X_test, y_test)
        
        # Obtém o nome do modelo e seus parâmetros
        model_name = algorithm.__class__.__name__
        model_params = algorithm.get_params()
        
        return score, model_name, model_params
