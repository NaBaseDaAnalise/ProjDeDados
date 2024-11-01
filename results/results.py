import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def visualize_grid_search(grid_search_results, model_name, target):   
    # Cria um DataFrame a partir dos resultados do grid search
    results_df = pd.DataFrame(grid_search_results)

    # Calcular a acurácia média geral
    overall_mean_score = results_df['mean_test_score'].mean()

    # Dicionário para armazenar os ganhos de acurácia para cada hiperparâmetro
    importance_scores = {}

    # Iterar sobre os resultados para calcular a média de acurácia por hiperparâmetro
    for index, row in results_df.iterrows():
        params = row['params']
        score = row['mean_test_score']
        
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
            importance_scores[param_name][param_value].append(score)

    # Calcular média de acurácia e ganho para cada hiperparâmetro
    for param_name, values in importance_scores.items():
        for param_value, scores in values.items():
            mean_score = sum(scores) / len(scores)
            gain = mean_score - overall_mean_score
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
    plt.title(f'Resultado da Partida: Importância dos Hiperparâmetros na Acurácia do Modelo {model_name}' if target == "classification" else f'{target.upper()}: Importância dos Hiperparâmetros no RMSE do Modelo {model_name}')
    plt.xlabel('Ganho de Acurácia' if target == "classification" else 'Ganho de RMSE')
    plt.ylabel('Hiperparâmetros')
    plt.savefig(f"results/classification/{model_name}_gsearch" if target == "classification" else f"results/{target}/{model_name}_gsearch")
    # plt.show()

def show_gsearch(path, target):
    if target == "classification":
        model_names = ["RandomForest","KNN","NeuralNetwork","GBT","LogisticRegression","AdaBoost","ExtraTrees","XGBoost","CatBoost"]
    else:
        model_names = ["SVR","RandomForest","NeuralNetwork","GBT","LinearRegression","Ridge","Lasso","AdaBoost","ExtraTrees","XGBoost","CatBoost"]

    for name in model_names:  
        gsearch_file = f'{path}{name.lower()}_grid_search.json'
        
        try:
            with open(gsearch_file, 'r') as f:
                gsearch_results = json.load(f)
                
            print(f"Visualizando resultados do Grid Search para {name}")
            visualize_grid_search(gsearch_results, name, target)
            
        except FileNotFoundError:
            print(f"Arquivo de resultados de Grid Search não encontrado para {name}.")

def show_best_model_metrics_classification():
    with open("modelagem/metrics/best_model_classification.json", 'r') as f:
        metrics_data = json.load(f)
    
    # Extração das métricas agregadas (ponderadas)
    accuracy = metrics_data["test_accuracy"]
    precision = metrics_data["classification_report"]["weighted avg"]["precision"]
    recall = metrics_data["classification_report"]["weighted avg"]["recall"]
    f1_score = metrics_data["classification_report"]["weighted avg"]["f1-score"]

    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_values = [accuracy, precision, recall, f1_score]

    # Configuração do gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics_names, metrics_values, color=['#9C27B0', '#00BCD4', '#FF9800', '#795548'])

    # Adiciona rótulos e título
    ax.set_xlabel("Métricas")
    ax.set_ylabel("Valores")
    ax.set_title(f"Métricas Agregadas do Melhor Modelo ({metrics_data['model_name']})")
    ax.set_ylim(0, 1 if max(metrics_values) <= 1 else max(metrics_values) * 1.1)  # Ajusta o limite y para métricas proporcionais

    # Adiciona valores acima das barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Salva o gráfico em um arquivo
    plt.tight_layout()
    plt.savefig('results/classification/overall_metrics.png')
    # plt.show()
        
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
    plt.title("Classificação: Comparação de Acurácia Média dos Modelos (Validação Cruzada vs. Conjunto de Teste)")
    plt.xticks(ticks=x, labels=model_names, rotation=45)
    plt.legend(loc="lower left", bbox_to_anchor=(0, 0))  # (x, y) para posicionar a legenda na parte inferior esquerda

    # Show the plot
    plt.tight_layout()
    plt.savefig('results/classification/model_comparsion.png')
    # plt.show()


def show_best_model_metrics_regression():
    with open("modelagem/metrics/handicap_metrics.json", 'r') as f:
        metrics_data = json.load(f)
    
    # Extração das métricas agregadas (ponderadas)
    RMSE_test = metrics_data[10]["RMSE_test"]
    RMSE_CV_mean = metrics_data[10]["RMSE_CV_mean"]
    MAE_test = metrics_data[10]["MAE_test"]
    MAE_CV_mean = metrics_data[10]["MAE_CV_mean"]
    R2_test = metrics_data[10]["R2_test"]
    R2_CV_mean = metrics_data[10]["R2_CV_mean"]
   


    metrics_names = ["RMSE_test", "RMSE_CV_mean", "MAE_test", "MAE_CV_mean","R2_test","R2_CV_mean"]
    metrics_values = [RMSE_test, RMSE_CV_mean, MAE_test, MAE_CV_mean,R2_test,R2_CV_mean]

    # Configuração do gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics_names, metrics_values, color=['#39e667', '#1f8239', '#3763e6', '#213e91','#eb314d', '#822331'])

    # Adiciona rótulos e título
    ax.set_xlabel("Métricas")
    ax.set_ylabel("Valores")
    ax.set_title(f"Total_Points: Métricas Agregadas do Melhor Modelo (CatBoost)")
    ax.set_ylim(0, 1 if max(metrics_values) <= 1 else max(metrics_values) * 1.1)  # Ajusta o limite y para métricas proporcionais

    # Adiciona valores acima das barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Salva o gráfico em um arquivo
    plt.tight_layout()
    plt.savefig('results/total_points/overall_metrics.png')
    # plt.show()
    
def show_regression(target):
    # Load metrics from JSON file
    metrics_file = 'modelagem/metrics/handicap_metrics.json' if target == "handicap" else 'modelagem/metrics/total_points_metrics.json'
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Extract model names, mean CV accuracies, and test accuracies
    model_names = [m["Model"] for m in metrics]
    mean_rmse = [m["RMSE_CV_mean"] for m in metrics]
    test_rmse = [m["RMSE_test"] for m in metrics]

    # Plotting
    x = np.arange(len(model_names))  # Label locations
    width = 0.35  # Width of the bars

    plt.figure(figsize=(14, 8))
    # Blue bars for cross-validation accuracies
    plt.bar(x - width/2, mean_rmse, width, label='Acurácia VC', color='skyblue')
    # Green bars for test set accuracies
    plt.bar(x + width/2, test_rmse, width, label='Acurácia Teste', color='lightgreen')

    # Labeling the plot
    plt.xlabel("Modelos")
    plt.ylabel("RMSE")
    plt.title(f"{target.upper()}: Comparação de RMSE Média dos Modelos (Validação Cruzada vs. Conjunto de Teste)")
    plt.xticks(ticks=x, labels=model_names, rotation=45)
    plt.legend(loc="lower left", bbox_to_anchor=(0, 0))  # (x, y) para posicionar a legenda na parte inferior esquerda

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'results/{target}/model_comparsion.png')
    # plt.show()        

def plot_results():
    show_classification()
    show_best_model_metrics_classification()
    show_gsearch("modelagem/result_gsearch/","classification")

    show_regression("total_points")
    show_gsearch("modelagem/total_points_gsearch/","total_points")
    
    show_regression("handicap")
    show_gsearch("modelagem/handicap_gsearch/","handicap")
    show_best_model_metrics_regression()
