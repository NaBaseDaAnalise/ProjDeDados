import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def execute_model(model_name, cv, X_train_scaled, X_test_scaled, y_train, y_test, search_best_params=False):
    models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'NeuralNetwork': MLPClassifier(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GBT': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'AdaBoost': AdaBoostClassifier(algorithm="SAMME",random_state=42),
        'ExtraTrees': ExtraTreesClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(random_state=42, silent=True)
    }

    if model_name not in models:
        raise ValueError(f"Modelo {model_name} não suportado. Escolha entre: {', '.join(models.keys())}")

    model = models[model_name]
    param_file = f'modelagem/result_params/{model_name.lower()}_best_params.json'

    if not search_best_params and os.path.exists(param_file):
        with open(param_file, 'r') as f:
            best_params = json.load(f)
        model.set_params(**best_params)
    
    else:
        if model_name == 'KNN':
            param_grid = {
                'n_neighbors': [9, 13, 15, 20],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
        elif model_name == 'NeuralNetwork':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu'],
                'solver': ['sgd'],
                'learning_rate': ['adaptive'],
                'max_iter': [300, 500, 1000],
                'learning_rate_init': [0.001, 0.01]
            }
        elif model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [300, 500],
                'max_depth': [15, 20, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 4, 8],
                'bootstrap': [True]
            }
        elif model_name == 'GBT':
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01],
                'max_depth': [3, 4, 10, None],
                'subsample': [1.0],
                'min_samples_split': [5],
                'min_samples_leaf': [1, 2]
            }
        elif model_name == 'LogisticRegression':
            # Definindo os parâmetros que serão utilizados
            param_grid = {
                'penalty': ['elasticnet', None],
                'C': [0.1, 1],
                'solver': ['saga'],
                'l1_ratio': [0.3, 0.5, 0.7]
            }

            # Filtrando combinações válidas
            valid_param_grid = []
            for penalty in param_grid['penalty']:
                if penalty == 'l2':
                    # 'liblinear', 'newton-cg', e 'sag' suportam 'l2'
                    for C_value in param_grid['C']:
                        for solver in ['liblinear', 'newton-cg', 'sag']:
                            valid_param_grid.append({'penalty': [penalty], 'C': [C_value], 'solver': [solver]})
                elif penalty == 'l1':
                    # 'liblinear' e 'saga' suportam 'l1'
                    for C_value in param_grid['C']:
                        for solver in ['liblinear', 'saga']:
                            valid_param_grid.append({'penalty': [penalty], 'C': [C_value], 'solver': [solver]})
                elif penalty == 'elasticnet':
                    # Apenas 'saga' suporta 'elasticnet'
                    for C_value in param_grid['C']:
                        for l1_ratio_value in [0.1, 0.5, 0.9]:  # Adicionando valores para 'l1_ratio'
                            valid_param_grid.append({
                                'penalty': [penalty],
                                'C': [C_value],
                                'solver': ['saga'],
                                'l1_ratio': [l1_ratio_value]
                            })
                elif penalty == 'none':
                    # 'saga' e 'newton-cg' suportam 'none'
                    for solver in ['saga', 'newton-cg']:
                        valid_param_grid.append({'penalty': [penalty], 'solver': [solver]})

            # Convertendo para o formato esperado pelo GridSearchCV
            param_grid = []
            for item in valid_param_grid:
                if item['penalty'] == ['elasticnet']:
                    param_grid.append({'penalty': item['penalty'], 'C': item['C'], 'solver': item['solver'], 'l1_ratio': item['l1_ratio']})
                elif 'C' in item:
                    param_grid.append({'penalty': item['penalty'], 'C': item['C'], 'solver': item['solver']})
                else:
                    param_grid.append({'penalty': item['penalty'], 'solver': item['solver']})



        elif model_name == 'AdaBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.5, 1.0,1.5]
            }
        elif model_name == 'ExtraTrees':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [13,15, 17, None],
                'min_samples_split': [4, 5, 9],
                'min_samples_leaf': [1],
                'bootstrap': [False]
            }
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [2, 3, 7],
                'subsample': [1.0],
                'colsample_bytree': [0.6, 0.8]
            }
        elif model_name == 'CatBoost':
            param_grid = {
                'iterations': [250, 500],
                'learning_rate': [0.01],
                'depth': [2, 4, 6],
                'l2_leaf_reg': [5, 7]
            }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train_scaled, y_train)
        # visualize_grid_search(grid_search)
        best_params = grid_search.best_params_
        
        # Convert cv_results_ to a JSON serializable dictionary
        cv_results_serializable = {}
        for key, value in grid_search.cv_results_.items():
            # Convert numpy arrays to lists
            if isinstance(value, np.ndarray):
                cv_results_serializable[key] = value.tolist()
            else:
                cv_results_serializable[key] = value

        # Save the converted cv_results_ dictionary to JSON
        with open(f"modelagem/result_gsearch/{model_name.lower()}_grid_search.json", 'w') as f:
            json.dump(cv_results_serializable, f, indent=4)
            
        with open(param_file, 'w') as f:
            json.dump(best_params, f)
            
        # Extrai os parâmetros e a acurácia média do grid_search.cv_results_
        params_list = grid_search.cv_results_['params']
        mean_scores = grid_search.cv_results_['mean_test_score']
        
        
        model.set_params(**best_params)

    return (model, best_params, params_list, mean_scores)

def visualize_grid_search(grid_search):
    # Obter os resultados da grid search
    results = grid_search.cv_results_
    
    # Extrair as acurácias e os parâmetros
    mean_test_scores = results['mean_test_score']
    params = results['params']

    # Plotar os resultados para visualização
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(mean_test_scores)), mean_test_scores, color='blue')
    plt.yticks(range(len(mean_test_scores)), [str(param) for param in params])
    plt.xlabel('Acurácia média')
    plt.title('Resultados do Grid Search para SVM')
    plt.show()

def save_metrics(model_names, mean_accuracies, test_accuracies, best_params):
    # Collect metrics into a list of dictionaries
    metrics = []
    for i, model_name in enumerate(model_names):
        metrics.append({
            "Model": model_name,
            "Mean_CV_Accuracy": mean_accuracies[i],
            "Test_Accuracy": test_accuracies[i],
            "Best_Parameters": best_params[i]
        })

    # Save metrics to a JSON file
    metrics_file = 'modelagem/metrics/classification_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Métricas salvas no arquivo {metrics_file}.")

def predict_and_metrics(X_train_scaled, X_test_scaled, y_train, y_test, experimentation_plan_data, search_best_params):
    # Configuração da validação cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_names = ["RandomForest","KNN","SVM","NeuralNetwork","GBT","LogisticRegression","AdaBoost","ExtraTrees","XGBoost","CatBoost"]
    models = []
    best_params = []
    mean_accuracies = []
    test_accuracies = []
    best_model_metrics = {}
    best_model_accuracy = 0
    best_model = KNeighborsClassifier()
    
    print("\n\n----------------------------------------------\n")
    print("Classificação para resultado da partida: \n")
    
    for name in model_names:
        (model, model_best_params, params_list, mean_scores) = execute_model(name,cv, X_train_scaled, X_test_scaled, y_train, y_test, search_best_params)
        models.append(model)
        best_params.append(model_best_params)
    
        # Prepara uma lista de dicionários para o DataFrame
        for params, score in zip(params_list, mean_scores):
            experimentation_plan_data.append({
                "Target": "Resultado da Partida",
                "Modelo": name,
                "Parâmetros": params,
                "Score da Validação Cruzada (Acurácia ou RMSE)": score
            })

    # Loop para treinar e avaliar os modelos
    for i, model in enumerate(models):
        score = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        mean_accuracy = score.mean()
        std_accuracy = score.std()
        print(f"{model_names[i]} - Acurácia média com validação cruzada (treino): {mean_accuracy:.4f}")
        print(f"{model_names[i]} - Desvio padrão da acurácia (treino): {std_accuracy:.4f}")

        # Avaliação no conjunto de teste
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{model_names[i]} - Acurácia no conjunto de teste:", test_accuracy)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)  # Gera o relatório como dicionário
        print(f"\n{model_names[i]} - Relatório de classificação no conjunto de teste:\n", classification_report(y_test, y_pred))

        # Atualiza métricas se for o melhor modelo até agora
        if mean_accuracy > best_model_accuracy:
            best_model_accuracy = mean_accuracy
            best_model = model
            best_model_metrics = {
                "model_name": model_names[i],
                "train_mean_accuracy": mean_accuracy,
                "train_std_accuracy": std_accuracy,
                "test_accuracy": test_accuracy,
                "classification_report": classification_rep
            }

        # Salva as métricas em listas
        mean_accuracies.append(mean_accuracy)
        test_accuracies.append(test_accuracy)
        
    y_pred_best = best_model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred_best)

    # Configuração do gráfico da matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Matriz de Confusão - Melhor Modelo: {best_model_metrics['model_name']}")

    # Salva o gráfico
    plt.tight_layout()
    plt.savefig("results/best_model_confusion_matrix.png")
    
    # Salva o dicionário de métricas do melhor modelo em um arquivo JSON
    with open('modelagem/metrics/best_model_classification.json', 'w') as json_file:
        json.dump(best_model_metrics, json_file, indent=4)
    save_metrics(model_names,mean_accuracies,test_accuracies, best_params)

def classification(experimentation_plan_data, search_best_params):
    df = pd.read_csv('pre_processamento/games_data_preproc.csv').copy()

    # Remover colunas irrelevantes
    df.drop(["Tm", "Opp"], axis=1, inplace=True)
    
    df['Resultado'] = df['Resultado'].map({'W': 1, 'L': 0})

    # Separação entre treino e teste
    X = df.drop(columns=["Resultado"])
    y = df["Resultado"] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalizar as features (padronização: média 0, desvio padrão 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    predict_and_metrics(X_train_scaled, X_test_scaled, y_train, y_test, experimentation_plan_data, search_best_params)

    # if hasattr(best_model, "predict_proba"):
    #     y_pred = best_model.predict(X_test_scaled)
    #     y_pred_proba = best_model.predict_proba(X_test_scaled)
    #     # Exemplo de como exibir as probabilidades de vitória (classe 1) e derrota (classe 0)
    #     for i in range(len(y_pred)):
    #         print(f"Exemplo {i+1} - Probabilidade de vitória: {y_pred_proba[i][1]:.4f}, Probabilidade de derrota: {y_pred_proba[i][0]:.4f}")
    # else:
    #     print(f"O modelo melohr modelo não suporta predição de probabilidade.")