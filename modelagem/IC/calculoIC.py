import pandas as pd
import math

# Função para calcular intervalo de confiança
def calculate_confidence_interval(mean, std, n, Z):
    margin_error = Z * (std / math.sqrt(n))
    lower = mean - margin_error
    upper = mean + margin_error
    return lower, upper

# Lista de arquivos de entrada e seus respectivos arquivos de saída
files = [
    {"input": "../plano_experimentacao/total_points_plan.csv", "output": "totalPoints_intervals_by_model.csv"},
    {"input": "../plano_experimentacao/classification_plan.csv", "output": "classification_intervals_by_model.csv"},
    {"input": "../plano_experimentacao/handicap_plan.csv", "output": "handicap_intervals_by_model.csv"}
]

# Configurações
###AJEItAR CONFIGURACOES DA VALDAÇÃO CRUZADA?? - HANNAH
n_folds = 5  # Número de folds na validação cruzada
Z = 1.96  # Valor crítico para 95% de confiança

# Processar cada arquivo
for file in files:
    input_path = file["input"]
    output_path = file["output"]

    # Carregar os dados do plano de experimentação
    print(f"Lendo o arquivo: {input_path}")
    experiment_data = pd.read_csv(input_path)

    # Remover espaços extras e normalizar os nomes das colunas
    experiment_data.columns = experiment_data.columns.str.strip()

    # Converter a coluna de métricas para valores numéricos
    experiment_data['Score da Validação Cruzada (Acurácia ou RMSE)'] = pd.to_numeric(
        experiment_data['Score da Validação Cruzada (Acurácia ou RMSE)'], errors='coerce'
    )

    # Agrupar por modelo e calcular estatísticas
    grouped_data = experiment_data.groupby('Modelo').agg(
        mean_score=('Score da Validação Cruzada (Acurácia ou RMSE)', 'mean'),
        std_score=('Score da Validação Cruzada (Acurácia ou RMSE)', 'std'),
        count=('Score da Validação Cruzada (Acurácia ou RMSE)', 'count')
    ).reset_index()

    # Calcular intervalos de confiança
    results = []
    for _, row in grouped_data.iterrows():
        model_name = row['Modelo']
        mean = row['mean_score']
        std = row['std_score']
        n = row['count']
        
        # Ignorar modelos sem dados suficientes
        if pd.isna(mean) or pd.isna(std) or n < 2:
            print(f"Modelo {model_name} ignorado por falta de dados suficientes.")
            continue

        # Calcular intervalo de confiança
        lower, upper = calculate_confidence_interval(mean, std, n, Z)
        results.append({
            "Model": model_name.strip(),
            "Mean": mean,
            "Lower Bound (95%)": lower,
            "Upper Bound (95%)": upper
        })

    # Converter os resultados para DataFrame
    results_df = pd.DataFrame(results)

    # Depuração: Exibir os resultados calculados
    print(f"Resultados calculados para {input_path}:")
    print(results_df)

    # Salvar os resultados em um arquivo CSV
    results_df.to_csv(output_path, index=False)
    print(f"Arquivo gerado com sucesso em: {output_path}")
