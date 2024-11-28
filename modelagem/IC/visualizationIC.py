import pandas as pd
import matplotlib.pyplot as plt

# Lista de arquivos de entrada e seus respectivos nomes de saída de visualização
files = [
    {"input": "totalPoints_intervals_by_model.csv", "output": "totalPoints_intervals_plot.png"},
    {"input": "classification_intervals_by_model.csv", "output": "classification_intervals_plot.png"},
    {"input": "handicap_intervals_by_model.csv", "output": "handicap_intervals_plot.png"}
]

# Função para criar visualizações
def create_visualizations(input_file, output_file):
    # Carregar o arquivo CSV
    df = pd.read_csv(input_file)

    # Verificar se o DataFrame está vazio
    if df.empty:
        print(f"O arquivo {input_file} está vazio. Nenhuma visualização foi criada.")
        return

    # Filtrar modelos com valores extremos (por exemplo, média muito alta)
    filtered_df = df[abs(df['Mean']) < 1e6]  # Ignora valores extremos maiores que 1 milhão
    if len(filtered_df) != len(df):
        print(f"Valores extremos removidos em {input_file}. Modelos removidos: {set(df['Model']) - set(filtered_df['Model'])}")

    # Criar gráfico de barras com intervalos de confiança
    plt.figure(figsize=(10, 6))
    plt.bar(
        filtered_df['Model'], 
        filtered_df['Mean'], 
        yerr=[filtered_df['Mean'] - filtered_df['Lower Bound (95%)'], filtered_df['Upper Bound (95%)'] - filtered_df['Mean']], 
        capsize=5, 
        color='skyblue'
    )
    plt.xlabel('Modelos')
    plt.ylabel('Média com Intervalos de Confiança')
    plt.title(f'Intervalos de Confiança - {input_file}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Salvar o gráfico em um arquivo
    plt.savefig(output_file)
    plt.close()
    print(f"Visualização criada e salva em: {output_file}")

# Gerar visualizações para cada arquivo
for file in files:
    create_visualizations(file["input"], file["output"])
