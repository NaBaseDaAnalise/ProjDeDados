import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
df = pd.read_csv('games_data_preproc.csv').copy()

# Separar as colunas que não devem ser incluídas no PCA
columns_to_exclude = ["Tm", "Opp", "Resultado"]
df_excluded = df[columns_to_exclude]  # Mantém as colunas excluídas
df_pca = df.drop(columns=columns_to_exclude)  # Colunas para o PCA

# Padronizar os dados (PCA assume que os dados estão padronizados)
scaler = StandardScaler()
df_pca_scaled = scaler.fit_transform(df_pca)

n_components = 10
# Executar o PCA (vamos usar 2 componentes como exemplo, mas pode ajustar conforme necessário)
pca = PCA(n_components=n_components)
pca_values = pca.fit_transform(df_pca_scaled)

# Criar um DataFrame com os resultados do PCA
column_names = [f'PCA{i+1}' for i in range(n_components)]  # Gera 'PCA1', 'PCA2', etc.

# Criar um DataFrame com os resultados do PCA
df_pca_result = pd.DataFrame(pca_values, columns=column_names)
# Concatenar com as colunas excluídas
df_final = pd.concat([df_excluded.reset_index(drop=True), df_pca_result], axis=1)

# Salvar o novo dataset com o PCA aplicado
df_final.to_csv('games_data_preproc_2.csv', index=False)
