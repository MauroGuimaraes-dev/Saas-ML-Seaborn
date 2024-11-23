# Importando as bibliotecas necessárias
import seaborn as sns  # Para carregar datasets de exemplo e criar visualizações
import pandas as pd    # Para manipulação de dados em DataFrames
import numpy as np     # Para operações numéricas
from sklearn.model_selection import train_test_split  # Para dividir dados em treino e teste

class DataModel:
    """
    Classe responsável por gerenciar os dados da aplicação.
    Fornece métodos para carregar datasets, preparar dados e identificar colunas numéricas.
    """
    
    @staticmethod
    def get_available_datasets():
        """
        Retorna um dicionário com os datasets disponíveis do Seaborn.
        Cada dataset é carregado diretamente da biblioteca Seaborn.
        """
        return {
            "Tips": sns.load_dataset("tips"),        # Dataset de gorjetas de restaurante
            "Iris": sns.load_dataset("iris"),        # Dataset clássico de flores Iris
            "Diamonds": sns.load_dataset("diamonds"), # Dataset de preços de diamantes
            "Penguins": sns.load_dataset("penguins") # Dataset de medidas de pinguins
        }
    
    @staticmethod
    def prepare_data(df):
        """
        Prepara os dados para treinamento do modelo.
        Args:
            df (pandas.DataFrame): DataFrame com os dados a serem preparados
        Returns:
            tuple: Dados divididos em treino e teste, ou None se não houver colunas numéricas suficientes
        """
        # Obtém todas as colunas numéricas do DataFrame
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Verifica se há pelo menos duas colunas numéricas
        if len(numeric_cols) >= 2:
            # Separa features (X) e target (y)
            X = df[numeric_cols[:-1]]  # Todas as colunas numéricas exceto a última
            y = df[numeric_cols[-1]]   # Última coluna numérica como target
            
            # Divide os dados em conjuntos de treino e teste
            return train_test_split(X, y, test_size=0.2, random_state=42)
        return None
    
    @staticmethod
    def get_numeric_columns(df):
        """
        Retorna as colunas numéricas do DataFrame.
        Args:
            df (pandas.DataFrame): DataFrame para extrair colunas numéricas
        Returns:
            pandas.Index: Lista de nomes das colunas numéricas
        """
        return df.select_dtypes(include=[np.number]).columns
