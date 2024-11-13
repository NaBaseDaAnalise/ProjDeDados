import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_handicap(df):
    if 'Tm' in df.columns and 'Opp' in df.columns:
        df['Handicap'] = df['Tm'] - df['Opp']
    else:
        raise ValueError("As colunas 'Tm' e 'Opp' não estão presentes no DataFrame.")
    
    return df

def calculate_total_points(df):
    if 'Tm' in df.columns and 'Opp' in df.columns:
        df['Total Points'] = df['Tm'] + df['Opp']
    else:
        raise ValueError("As colunas 'Tm' e 'Opp' não estão presentes no DataFrame.")
    
    return df

def execute_model(model_name, cv, X_train_scaled, X_test_scaled, y_train, y_test, target, search_best_params=False):
    models = {
        'KNN': KNeighborsRegressor(),
        'SVR': SVR(),
        'NeuralNetwork': MLPRegressor(max_iter=1000, random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GBT': GradientBoostingRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, silent=True)
    }

    if model_name not in models:
        raise ValueError(f"Modelo {model_name} não suportado. Escolha entre: {', '.join(models.keys())}")

    model = models[model_name]
    param_file = f'modelagem/{target}_params/{model_name.lower()}_best_params.json'

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
        elif model_name == 'SVR':
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
        elif model_name == 'LinearRegression':
            param_grid = {
                'fit_intercept': [True, False],      # Se deve calcular o intercepto ou não
                'copy_X': [True, False],             # Se deve copiar os dados de entrada ou sobrescrevê-los
            }
        elif model_name == 'Ridge':
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            }
        elif model_name == 'Lasso':
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10],
                'max_iter': [1000, 2000, 5000]
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

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(X_train_scaled, y_train)
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
        with open(f"modelagem/{target}_gsearch/{model_name.lower()}_grid_search.json", 'w') as f:
            json.dump(cv_results_serializable, f, indent=4)

            
        with open(param_file, 'w') as f:
            json.dump(best_params, f)

        model.set_params(**best_params)
    
    return (model, best_params)

def predict_and_metrics(X_train_scaled, X_test_scaled, y_train, y_test, search_best_params, target):
    # Configuração da validação cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_names = ["SVR","RandomForest","NeuralNetwork","GBT","LinearRegression","Ridge","Lasso","AdaBoost","ExtraTrees","XGBoost","CatBoost"]
    models = []
    params = []
    models_metrics = []
    best_model_rmse = 10000000
    best_model = RandomForestRegressor()
    
    # Definição dos scorers para RMSE, MAE e R²
    rmse_scorer = make_scorer(root_mean_squared_error) 
    mae_scorer = make_scorer(mean_absolute_error)
    r2_scorer = make_scorer(r2_score)
    
    print("\n\n----------------------------------------------\n")
    print(f"Regressão para {target}: \n")
    
    for name in model_names:
        (model, best_params) = execute_model(name,cv, X_train_scaled, X_test_scaled, y_train, y_test,target,search_best_params=search_best_params)
        models.append(model)
        params.append(best_params)

    # Loop para treinar e avaliar os modelos
    for i, model in enumerate(models):
        rmse_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=rmse_scorer)
        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()
        
        # Média e desvio padrão do MAE
        mae_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=mae_scorer)
        mean_mae = mae_scores.mean()
        std_mae = mae_scores.std()
        
        # Média e desvio padrão do R²
        r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=r2_scorer)
        mean_r2 = r2_scores.mean()
        std_r2 = r2_scores.std()
            
        # Exibir os resultados
        print(f"{model_names[i]} - RMSE médio com validação cruzada (treino): {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"{model_names[i]} - MAE médio com validação cruzada (treino): {mean_mae:.4f} ± {std_mae:.4f}")
        print(f"{model_names[i]} - R² médio com validação cruzada (treino): {mean_r2:.4f} ± {std_r2:.4f}")
        
        # Avaliação no conjunto de teste
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{model_names[i]} - Métricas de regressão no conjunto de teste:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² score: {r2:.4f}")
        
        metrics = {
            "Model": model_names[i],
            "RMSE_test": rmse,
            "MAE_test": mae,
            "R2_test": r2,
            "RMSE_CV_mean": mean_rmse,
            "RMSE_CV_std": std_rmse,
            "MAE_CV_mean": mean_mae,
            "MAE_CV_std": std_mae,
            "R2_CV_mean": mean_r2,
            "R2_CV_std": std_r2,
            "Best_Parameters": params[i]
        }
        
        # Atualiza métricas se for o melhor modelo até agora
        if mean_rmse < best_model_rmse:
            best_model_rmse = mean_rmse
            best_model = model

        # Salva as métricas em listas
        models_metrics.append(metrics)
        
    metrics_file = f'modelagem/metrics/{target}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(models_metrics, f, indent=4)
        
    print(f"Métricas salvas no arquivo {metrics_file}.")
    
def regresssion(search_best_params, target):
    df = pd.read_csv('pre_processamento/games_data_preproc_final.csv').copy()
    
    if target == "handicap":
        df_handicap = calculate_handicap(df)

        # Remover colunas irrelevantes
        df_handicap.drop(["Resultado", "Tm", "Opp"], axis=1, inplace=True)
        
        # Separação entre treino e teste
        X = df_handicap.drop(columns=["Handicap"])
        y = df_handicap["Handicap"] 
        
    if target == "total_points":
        df_total_points = calculate_total_points(df)
        df_total_points.drop(["Resultado", "Tm", "Opp"], axis=1, inplace=True)

        # Separação entre treino e teste
        X = df_total_points.drop(columns=["Total Points"])
        y = df_total_points["Total Points"] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalizar as features (padronização: média 0, desvio padrão 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predict_and_metrics(X_train_scaled, X_test_scaled, y_train, y_test,search_best_params, target)
        