import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# Carregar o DataFrame
df = pd.read_csv('../data/games_data_preproc.csv').copy()

# Remover colunas irrelevantes
df.drop(["Date"], axis=1, inplace=True)

# Analisar e preencher valores faltantes apenas nas colunas numéricas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Converter as colunas categóricas 'Team' e 'Opponent' para valores numéricos
label_encoder = LabelEncoder()
df['Team'] = label_encoder.fit_transform(df['Team'])
df['Opponent'] = label_encoder.fit_transform(df['Opponent'])

# Dividir os dados em features (X) e target (y)
X = df.drop(columns=["Resultado"])  # Exclui a coluna "Resultado"
y = df["Resultado"]  # Define a coluna "Resultado" como alvo

# Dividir os dados em treino e teste (80% treino e 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar as features (padronização: média 0, desvio padrão 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configuração da validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Treinar o modelo SVM com validação cruzada usando apenas o conjunto de treino
svm_model = SVC()

# Realizar a validação cruzada apenas no conjunto de treino
scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

# Resultados da validação cruzada
print(f"Acurácia média com validação cruzada (treino): {scores.mean():.4f}")
print(f"Desvio padrão da acurácia (treino): {scores.std():.4f}")

# Treinamento final no conjunto de treino completo
svm_model.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred = svm_model.predict(X_test_scaled)

# Avaliar a performance no conjunto de teste
print("\nAcurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação no conjunto de teste:\n", classification_report(y_test, y_pred))

# Treinar o modelo Random Forest sem normalização (os dados numéricos não precisam ser normalizados para Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Fazer previsões e avaliar a performance no conjunto de teste
y_pred_rf = rf_model.predict(X_test)

# Avaliar a acurácia e o relatório de classificação do Random Forest
print("\nAcurácia com Random Forest:", accuracy_score(y_test, y_pred_rf))
print("\nRelatório de classificação com Random Forest:\n", classification_report(y_test, y_pred_rf))

# Opcional: Aplicar validação cruzada no modelo Random Forest
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"\nAcurácia média com validação cruzada (Random Forest - treino): {rf_scores.mean():.4f}")
print(f"Desvio padrão da acurácia (Random Forest - treino): {rf_scores.std():.4f}")