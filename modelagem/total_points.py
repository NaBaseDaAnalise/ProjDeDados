import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def calculate_total_points(df):
    if 'Tm' in df.columns and 'Opp' in df.columns:
        df['Total Points'] = df['Tm'] + df['Opp']
    else:
        raise ValueError("As colunas 'Tm' e 'Opp' não estão presentes no DataFrame.")
    
    return df

def execute_model(model_name, cv, X_train_scaled, X_test_scaled, y_train, y_test, search_best_params=False):
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
    param_file = f'modelagem/total_points_params/{model_name.lower()}_best_params.json'

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

        with open(param_file, 'w') as f:
            json.dump(best_params, f)

        model.set_params(**best_params)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n{model_name} - Métricas de regressão no conjunto de teste:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R² score: {r2_score(y_test, y_pred):.4f}")

def pred_total_points(search_best_params):
    df = pd.read_csv('pre_processamento/games_data_preproc.csv').copy()
    df = calculate_total_points(df)
    
    # Remover colunas irrelevantes
    df.drop(["Resultado", "Tm", "Opp"], axis=1, inplace=True)
    
    # Separação entre treino e teste
    X = df.drop(columns=["Total Points"])
    y = df["Total Points"] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalizar as features (padronização: média 0, desvio padrão 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Configuração da validação cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n\n----------------------------------------------\n")
    print("Regressão para pontuação total: \n")
    execute_model("SVR",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("RandomForest",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("KNN",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("NeuralNetwork",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("GBT",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("LinearRegression",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("Ridge",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("Lasso",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("AdaBoost",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("ExtraTrees",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("XGBoost",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)
    execute_model("CatBoost",cv, X_train_scaled, X_test_scaled, y_train, y_test,search_best_params=search_best_params)

