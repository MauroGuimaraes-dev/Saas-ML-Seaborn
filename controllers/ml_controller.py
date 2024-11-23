# Importando bibliotecas e módulos necessários
import streamlit as st                  # Para criar a interface web
from models.data_model import DataModel  # Modelo para gerenciamento de dados
from models.ml_model import MLModel      # Modelo para machine learning
from views.data_view import DataView     # View para interface do usuário

class MLController:
    """
    Controlador principal da aplicação.
    Coordena a interação entre os modelos e as views.
    """
    
    def __init__(self):
        """
        Inicializa o controlador criando instâncias dos modelos e views.
        """
        self.data_model = DataModel()     # Instância do modelo de dados
        self.ml_model = MLModel()         # Instância do modelo de ML
        self.view = DataView()            # Instância da view
        
    def initialize_page(self):
        """
        Configura a página inicial da aplicação.
        Define o título e layout da página.
        """
        # Configura o layout da página
        st.set_page_config(page_title="Máquina Preditiva", layout="wide")
        
        # Define o estilo CSS personalizado
        st.markdown("""
            <style>
            .title-container {
                background-color: #1E395B;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .main-title {
                background: linear-gradient(45deg, #4e9af1, #0056b3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5em;
                font-weight: bold;
                text-align: center;
                margin: 0;
                padding: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            .subtitle {
                color: #CCCCCC;
                text-align: center;
                font-size: 1.2em;
                margin: 5px 0 0 0;
                padding: 0;
            }
            </style>
            <div class="title-container">
                <h1 class="main-title">Máquina Preditiva com Datasets Seaborn</h1>
                <p class="subtitle">Desenvolvido por Mauro Guimarães</p>
            </div>
            """, unsafe_allow_html=True)
        
    def get_user_selections(self):
        """
        Obtém as seleções do usuário para dataset e algoritmo.
        Returns:
            tuple: Dataset selecionado, algoritmo selecionado, datasets disponíveis, algoritmos disponíveis
        """
        # Obtém listas de datasets e algoritmos disponíveis
        datasets = self.data_model.get_available_datasets()
        algorithms = self.ml_model.get_available_algorithms()
        
        # Cria layout com duas colunas
        col1, col2 = st.columns(2)
        
        # Cria caixas de seleção para dataset e algoritmo
        with col1:
            selected_dataset = st.selectbox("Selecione o Dataset:", list(datasets.keys()))
        with col2:
            selected_algorithm = list(algorithms.keys())[0]  # Pega o primeiro algoritmo como padrão
            algorithm_params = algorithms[selected_algorithm]  # Obtém os parâmetros do algoritmo
            
        # Adiciona uma seção para configuração do modelo
        st.subheader("Configuração do Modelo")
        selected_algorithm = st.selectbox("Selecione o Algoritmo:", list(algorithms.keys()))
            
        return selected_dataset, selected_algorithm, datasets, algorithms

    def process_data(self, df, dataset_name, model_name):
        """
        Processa o dataset selecionado.
        Args:
            df (pandas.DataFrame): DataFrame a ser processado
            dataset_name (str): Nome do dataset selecionado
            model_name (str): Nome do modelo selecionado
        Returns:
            tuple: Dados processados e colunas numéricas
        """
        # Obtém colunas numéricas do dataset
        numeric_cols = self.data_model.get_numeric_columns(df)
        
        # Exibe informações e visualizações do dataset
        self.view.show_dataset_info(df)
        self.view.show_visualizations(df, numeric_cols, dataset_name, model_name)
        
        # Prepara os dados para treinamento
        data = self.data_model.prepare_data(df)
        return data, numeric_cols
    
    def train_model(self, data, algorithm):
        """
        Treina e avalia o modelo selecionado.
        Args:
            data: Dados preparados para treinamento
            algorithm: Algoritmo selecionado para treinamento
        """
        if data is not None:
            # Desempacota os dados de treino e teste
            X_train, X_test, y_train, y_test = data
            # Treina o modelo e obtém o score e informações do modelo
            score, model_name, model_params = self.ml_model.train_and_evaluate(algorithm, X_train, X_test, y_train, y_test)
            # Exibe o resultado com informações do modelo
            self.view.show_model_result(score, model_name, model_params)
        else:
            # Exibe mensagem de erro se não houver dados suficientes
            self.view.show_error_message()

    def run_application(self):
        """
        Executa o fluxo principal da aplicação.
        """
        # Inicializa a página
        self.initialize_page()
        
        # Obtém seleções do usuário
        selected_dataset, selected_algorithm, datasets, algorithms = self.get_user_selections()
        
        # Obtém o DataFrame selecionado
        df = datasets[selected_dataset]
        
        # Processa os dados e obtém visualizações
        data, numeric_cols = self.process_data(df, selected_dataset, selected_algorithm)
        
        # Adiciona um botão para executar o modelo
        if st.button("Executar Modelo Preditivo", type="primary", use_container_width=True):
            # Obtém o algoritmo selecionado
            algorithm = algorithms[selected_algorithm]
            
            # Treina e avalia o modelo
            self.train_model(data, algorithm)
