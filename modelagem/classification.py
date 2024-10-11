import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt



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
    
def execute_GradientBoostedTrees(cv, X_train, X_test, y_train, y_test):
    gb_model = GradientBoostingClassifier(random_state=42)

    # Realizar a validação cruzada no conjunto de treino
    scores = cross_val_score(gb_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Gradient Boosted Trees - Acurácia média com validação cruzada (treino): {scores.mean():.4f}")
    print(f"Gradient Boosted Trees - Desvio padrão da acurácia (treino): {scores.std():.4f}")

    # Treinamento final no conjunto de treino completo
    gb_model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = gb_model.predict(X_test)
    print("\nGradient Boosted Trees - Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
    print("\nGradient Boosted Trees - Relatório de classificação no conjunto de teste:\n", classification_report(y_test, y_pred))

def execute_NeuralNetwork(cv, X_train_scaled, X_test_scaled, y_train, y_test, hidden_layer_sizes=(100,)):
    nn_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, random_state=42)

    # Realizar a validação cruzada no conjunto de treino
    scores = cross_val_score(nn_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"Rede Neural - Acurácia média com validação cruzada (treino): {scores.mean():.4f}")
    print(f"Rede Neural - Desvio padrão da acurácia (treino): {scores.std():.4f}")

    # Treinamento final no conjunto de treino completo
    nn_model.fit(X_train_scaled, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = nn_model.predict(X_test_scaled)
    print("\nRede Neural - Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
    print("\nRede Neural - Relatório de classificação no conjunto de teste:\n", classification_report(y_test, y_pred))

def execute_KNN(cv, X_train_scaled, X_test_scaled, y_train, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Realizar a validação cruzada no conjunto de treino
    scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"KNN - Acurácia média com validação cruzada (treino): {scores.mean():.4f}")
    print(f"KNN - Desvio padrão da acurácia (treino): {scores.std():.4f}")

    # Treinamento final no conjunto de treino completo
    knn_model.fit(X_train_scaled, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = knn_model.predict(X_test_scaled)
    print("\nKNN - Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
    print("\nKNN - Relatório de classificação no conjunto de teste:\n", classification_report(y_test, y_pred))

def execute_SVM(cv, X_train_scaled, X_test_scaled, y_train, y_test, search_best_params=False, param_file='best_svm_params.json'):
    svm_model = SVC()

    # Se search_best_params for False, carregar os parâmetros de um arquivo JSON existente
    if not search_best_params and os.path.exists(param_file):
        with open(param_file, 'r') as f:
            best_params = json.load(f)
        print(f"Carregando melhores parâmetros do arquivo {param_file}: {best_params}")
    
    else:
        # Definir os parâmetros a serem testados pelo GridSearchCV
        param_grid = {
            'C': [0.1, 1, 10, 100],  # Parâmetro de regularização
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Funções do kernel
            'gamma': ['scale', 'auto'],  # Coeficiente do kernel
        }

        # Configurar o GridSearchCV para procurar os melhores parâmetros com validação cruzada
        grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train_scaled, y_train)

        # Obter os melhores parâmetros
        best_params = grid_search.best_params_
        print("\nMelhores parâmetros encontrados pelo GridSearchCV:")
        print(best_params)

        # Salvar os melhores parâmetros em um arquivo JSON
        with open(param_file, 'w') as f:
            json.dump(best_params, f)
        print(f"Melhores parâmetros salvos no arquivo {param_file}")

    # Treinar o modelo final com os melhores parâmetros
    svm_model.set_params(**best_params)
    svm_model.fit(X_train_scaled, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = svm_model.predict(X_test_scaled)
    print("\nSVM com melhores parâmetros - Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
    print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

def execute_RF(cv, X_train, X_test, y_train, y_test, search_best_params=False, param_file='best_rf_params.json'):
    rf_model = RandomForestClassifier(random_state=42)

    # Se search_best_params for False, carregar os parâmetros de um arquivo JSON existente
    if not search_best_params and os.path.exists(param_file):
        with open(param_file, 'r') as f:
            best_params = json.load(f)
        print(f"Carregando melhores parâmetros do arquivo {param_file}: {best_params}")
    
    else:
        # Definir os parâmetros a serem testados pelo GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Configurar o GridSearchCV para procurar os melhores parâmetros com validação cruzada
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("\nMelhores parâmetros encontrados pelo GridSearchCV:")
        print(best_params)

        # Salvar os melhores parâmetros em um arquivo JSON
        with open(param_file, 'w') as f:
            json.dump(best_params, f)
        print(f"Melhores parâmetros salvos no arquivo {param_file}")

    # Treinar o modelo final com os melhores parâmetros
    rf_model.set_params(**best_params)
    rf_model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = rf_model.predict(X_test)
    print("\nRandom Forest com melhores parâmetros - Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
    print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))


df = pd.read_csv('../data/games_data_preproc.csv').copy()

# Remover colunas irrelevantes
df.drop(["Tm", "Opp"], axis=1, inplace=True)

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

# execute_SVM(cv, X_train_scaled, X_test_scaled, y_train, y_test)
execute_RF(cv, X_train, X_test, y_train, y_test)
execute_KNN(cv, X_train, X_test, y_train, y_test)
execute_GradientBoostedTrees(cv, X_train, X_test, y_train, y_test)
execute_NeuralNetwork(cv, X_train, X_test, y_train, y_test)