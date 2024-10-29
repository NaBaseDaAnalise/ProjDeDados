import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def visualize_grid_search(grid_search_results, model_name):   
    # Cria um DataFrame a partir dos resultados do grid search
    results_df = pd.DataFrame(grid_search_results)

    # Calcular a acurácia média geral
    overall_mean_accuracy = results_df['mean_test_score'].mean()

    # Dicionário para armazenar os ganhos de acurácia para cada hiperparâmetro
    importance_scores = {}

    # Iterar sobre os resultados para calcular a média de acurácia por hiperparâmetro
    for index, row in results_df.iterrows():
        params = row['params']
        accuracy = row['mean_test_score']
        
        # Para cada parâmetro na combinação
        for param_name, param_value in params.items():
            # Converte listas para tuplas para usar como chave
            if isinstance(param_value, list):
                param_value = tuple(param_value)

            if param_name not in importance_scores:
                importance_scores[param_name] = {}

            if param_value not in importance_scores[param_name]:
                importance_scores[param_name][param_value] = []

            # Adiciona a acurácia à lista correspondente
            importance_scores[param_name][param_value].append(accuracy)

    # Calcular média de acurácia e ganho para cada hiperparâmetro
    for param_name, values in importance_scores.items():
        for param_value, accuracies in values.items():
            mean_accuracy = sum(accuracies) / len(accuracies)
            gain = mean_accuracy - overall_mean_accuracy
            importance_scores[param_name][param_value] = gain

    # Criar DataFrame para os resultados
    importance_list = []
    for param_name, values in importance_scores.items():
        for param_value, gain in values.items():
            importance_list.append({'Hyperparameter': f'{param_name} = {param_value}', 'Gain': gain})

    importance_df = pd.DataFrame(importance_list)

    # Visualizar os resultados
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='Gain', y='Hyperparameter', palette='viridis')
    plt.axvline(0, color='red', linestyle='--')  # Linha para mostrar acurácia de referência
    plt.title(f'Importância dos Hiperparâmetros na Acurácia do Modelo {model_name}')
    plt.xlabel('Ganho de Acurácia')
    plt.ylabel('Hiperparâmetros')
    plt.show()

def show_gsearch():
    model_names = ["RandomForest","KNN","NeuralNetwork","GBT","LogisticRegression","AdaBoost","ExtraTrees","XGBoost","CatBoost"]

    for name in model_names:  
        gsearch_file = f'modelagem/result_gsearch/{name.lower()}_grid_search.json'
        
        try:
            with open(gsearch_file, 'r') as f:
                gsearch_results = json.load(f)
                
            print(f"Visualizando resultados do Grid Search para {name}")
            visualize_grid_search(gsearch_results, name)
            
        except FileNotFoundError:
            print(f"Arquivo de resultados de Grid Search não encontrado para {name}.")

def show_classification():
    # Load metrics from JSON file
    metrics_file = 'modelagem/metrics/classification_metrics.json'
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Extract model names, mean CV accuracies, and test accuracies
    model_names = [m["Model"] for m in metrics]
    mean_accuracies = [m["Mean_CV_Accuracy"] for m in metrics]
    test_accuracies = [m["Test_Accuracy"] for m in metrics]

    # Plotting
    x = np.arange(len(model_names))  # Label locations
    width = 0.35  # Width of the bars

    plt.figure(figsize=(14, 8))
    # Blue bars for cross-validation accuracies
    plt.bar(x - width/2, mean_accuracies, width, label='Acurácia VC', color='skyblue')
    # Green bars for test set accuracies
    plt.bar(x + width/2, test_accuracies, width, label='Acurácia Teste', color='lightgreen')

    # Labeling the plot
    plt.xlabel("Modelos")
    plt.ylabel("Acurácia")
    plt.title("Comparação de Acurácia Média dos Modelos (Validação Cruzada vs. Conjunto de Teste)")
    plt.xticks(ticks=x, labels=model_names, rotation=45)
    plt.legend(loc="lower left", bbox_to_anchor=(0, 0))  # (x, y) para posicionar a legenda na parte inferior esquerda

    # Show the plot
    plt.tight_layout()
    plt.show()
    

def plot_results():
    show_classification()
    show_gsearch()