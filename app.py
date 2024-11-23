# Importando o controlador principal
from controllers.ml_controller import MLController

def main():
    """
    Função principal que inicializa e executa a aplicação.
    """
    # Cria uma instância do controlador
    controller = MLController()
    
    # Executa a aplicação
    controller.run_application()

# Ponto de entrada da aplicação
if __name__ == "__main__":
    main()
