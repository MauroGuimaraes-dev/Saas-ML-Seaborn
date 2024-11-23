# Importando bibliotecas necessárias
import streamlit as st      # Para criar a interface web
import seaborn as sns      # Para criar visualizações
import matplotlib.pyplot as plt  # Para manipular gráficos

class DataView:
    """
    Classe responsável pela interface do usuário e visualização dos dados.
    Fornece métodos para exibir informações, gráficos e resultados do modelo.
    """
    
    @staticmethod
    def show_dataset_info(df):
        """
        Exibe informações gerais sobre o dataset.
        Args:
            df (pandas.DataFrame): DataFrame a ser analisado
        """
        # Cria duas colunas para o layout
        col1, col2 = st.columns(2)

        # Coluna 1: Amostra do Dataset
        with col1:
            st.subheader("Amostra do Dataset")
            st.dataframe(df.head())

        # Coluna 2: Informações do Dataset
        with col2:
            st.subheader("Informações do Dataset")
            # Cria um container estilizado para as informações
            info_html = f"""
            <style>
            .info-container {{
                background-color: #f0f2f6;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
            }}
            .info-item {{
                margin: 10px 0;
                padding: 8px;
                background-color: white;
                border-radius: 5px;
            }}
            .info-label {{
                color: #1E395B;
                font-weight: bold;
            }}
            .info-value {{
                font-size: 1.2em;
                color: #333;
            }}
            </style>
            <div class="info-container">
                <div class="info-item">
                    <span class="info-label">Número de registros:</span>
                    <div class="info-value">{df.shape[0]}</div>
                </div>
                <div class="info-item">
                    <span class="info-label">Número de variáveis:</span>
                    <div class="info-value">{df.shape[1]}</div>
                </div>
            </div>
            """
            st.markdown(info_html, unsafe_allow_html=True)
            
            # Mostra estatísticas descritivas do dataset
            st.subheader("Descrição Estatística")
            st.dataframe(df.describe())

    @staticmethod
    def show_visualizations(df, numeric_cols, dataset_name, model_name):
        """
        Cria e exibe visualizações dos dados.
        Args:
            df (pandas.DataFrame): DataFrame para criar visualizações
            numeric_cols (list): Lista de colunas numéricas
            dataset_name (str): Nome do dataset selecionado
            model_name (str): Nome do modelo selecionado
        """
        # Cria um cabeçalho estilizado com as informações de dataset e modelo
        header_html = f"""
        <style>
        .header-container {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 
                0 2px 15px rgba(0,0,0,0.1),
                inset 0 1px 2px rgba(255,255,255,0.5);
            border: 1px solid rgba(255,255,255,0.3);
        }}
        .header-title {{
            font-size: 1.6em;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            color: #1E395B;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
        }}
        .header-info {{
            display: flex;
            justify-content: space-around;
            padding: 10px 0;
            gap: 20px;
        }}
        .info-box {{
            flex: 1;
            background: linear-gradient(135deg, #ffffff 0%, #f0f2f6 100%);
            padding: 12px 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 
                0 2px 10px rgba(0,0,0,0.05),
                inset 0 1px 2px rgba(255,255,255,0.9);
            border: 1px solid rgba(255,255,255,0.5);
            transition: transform 0.2s ease;
        }}
        .info-box:hover {{
            transform: translateY(-2px);
            box-shadow: 
                0 4px 15px rgba(0,0,0,0.1),
                inset 0 1px 2px rgba(255,255,255,0.9);
        }}
        .info-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .info-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #1E395B;
            text-shadow: 1px 1px 1px rgba(255,255,255,0.8);
        }}
        </style>
        <div class="header-container">
            <div class="header-title">Análise de Dados</div>
            <div class="header-info">
                <div class="info-box">
                    <div class="info-label">Dataset</div>
                    <div class="info-value">{dataset_name}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Modelo</div>
                    <div class="info-value">{model_name}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
        
        if len(numeric_cols) > 0:
            # Cria layout com duas colunas para os gráficos
            col1, col2 = st.columns(2)

            with col1:
                # Cria histograma da primeira variável numérica
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.histplot(data=df, x=numeric_cols[0], ax=ax1)
                ax1.set_title(f'Distribuição de {numeric_cols[0]}')
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                # Cria gráfico de dispersão das duas primeiras variáveis numéricas
                if len(numeric_cols) >= 2:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=ax2)
                    ax2.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
                    st.pyplot(fig2)
                    plt.close(fig2)

    @staticmethod
    def show_model_result(score, model_name=None, model_params=None):
        """
        Exibe o resultado do modelo (R² score) e informações do modelo.
        Args:
            score (float): Valor do R² score do modelo
            model_name (str): Nome do modelo usado
            model_params (dict): Dicionário com os parâmetros do modelo
        """
        # Primeiro, define os estilos
        st.markdown("""
        <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        .metric-title {
            font-size: 1.8em;
            color: #1E395B;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 3em;
            font-weight: bold;
            color: #1E395B;
        }
        .metric-description {
            font-size: 1.2em;
            color: #666;
            margin-top: 10px;
        }
        .model-info {
            background-color: #e6e9ef;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            text-align: left;
        }
        .model-name {
            font-size: 1.4em;
            color: #1E395B;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .param-title {
            font-size: 1.2em;
            color: #1E395B;
            margin-top: 10px;
            margin-bottom: 5px;
        }
        .param-item {
            margin-left: 15px;
            color: #333;
            font-family: monospace;
        }
        </style>
        """, unsafe_allow_html=True)

        # Exibe o score do modelo
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Resultado do Modelo</div>
            <div class="metric-value">{score:.4f}</div>
            <div class="metric-description">Coeficiente de Determinação (R²)</div>
        </div>
        """, unsafe_allow_html=True)

        # Se tiver informações do modelo, exibe em um componente separado
        if model_name and model_params:
            st.markdown("""
            <div class="model-info">
                <div class="model-name">{}</div>
                <div class="param-title">Hiperparâmetros:</div>
                {}
            </div>
            """.format(
                model_name,
                "\n".join(f'<div class="param-item">• {k}: {v}</div>' for k, v in model_params.items())
            ), unsafe_allow_html=True)

    @staticmethod
    def show_error_message():
        """
        Exibe mensagem de erro quando não há dados suficientes.
        """
        st.warning("Dataset não possui variáveis numéricas suficientes para treinar o modelo.")
