import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

def execute_model(model_name, cv, X_train_scaled, X_test_scaled, y_train, y_test, search_best_params=False):
    models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
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
    param_file = f'modelagem/{model_name.lower()}_best_params.json'

    if not search_best_params and os.path.exists(param_file):
        with open(param_file, 'r') as f:
            best_params = json.load(f)
        model.set_params(**best_params)
    
    else:
        if model_name == 'KNN':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
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
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [300, 500, 1000, 2500],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        elif model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [7, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        elif model_name == 'GBT':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'LogisticRegression':
            param_grid = {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            }
        elif model_name == 'AdaBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0]
            }
        elif model_name == 'ExtraTrees':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [7, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        elif model_name == 'CatBoost':
            param_grid = {
                'iterations': [500, 1000],
                'learning_rate': [0.01, 0.1],
                'depth': [4, 6, 10],
                'l2_leaf_reg': [1, 3, 5, 7]
            }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train_scaled, y_train)
        best_params = grid_search.best_params_

        with open(param_file, 'w') as f:
            json.dump(best_params, f)

        model.set_params(**best_params)

    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"{model_name} - Acurácia média com validação cruzada (treino): {scores.mean():.4f}")
    print(f"{model_name} - Desvio padrão da acurácia (treino): {scores.std():.4f}")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{model_name} - Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
    print(f"\n{model_name} - Relatório de classificação no conjunto de teste:\n", classification_report(y_test, y_pred))

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

def classification(search_best_params):
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

    # Configuração da validação cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    execute_model("SVM",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("RandomForest",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("KNN",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("NeuralNetwork",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("GBT",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("LogisticRegression",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("AdaBoost",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("ExtraTrees",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("XGBoost",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("CatBoost",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
