import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca_team_stats(n_components):
    df = pd.read_csv('pre_processamento/games_data_preproc.csv').copy()

    if n_components != 0:

        # Carregar o dataset

        # Colunas para o PCA (Team e Opp separadamente)
        team_pca_columns = [
            'Previous_Team_Pace', 'Previous_Team_eFG%', 'Previous_Team_TOV%', 
            'Previous_Team_ORB%', 'Previous_Team_FT/FGA', 'Previous_Team_ORtg'
        ]
        opp_pca_columns = [
            'Previous_Opponent_Pace', 'Previous_Opponent_eFG%', 'Previous_Opponent_TOV%', 
            'Previous_Opponent_ORB%', 'Previous_Opponent_FT/FGA', 'Previous_Opponent_ORtg'
        ]
        
        # Padronizar e aplicar PCA nas colunas do time
        scaler_team = StandardScaler()
        df_team_scaled = scaler_team.fit_transform(df[team_pca_columns])
        pca_team = PCA(n_components=n_components)
        pca_team_values = pca_team.fit_transform(df_team_scaled)
        team_column_names = [f'PCA_Team_{i+1}' for i in range(n_components)]
        df_team_pca_result = pd.DataFrame(pca_team_values, columns=team_column_names)
        
        # Padronizar e aplicar PCA nas colunas do oponente
        scaler_opp = StandardScaler()
        df_opp_scaled = scaler_opp.fit_transform(df[opp_pca_columns])
        pca_opp = PCA(n_components=n_components)
        pca_opp_values = pca_opp.fit_transform(df_opp_scaled)
        opp_column_names = [f'PCA_Opp_{i+1}' for i in range(n_components)]
        df_opp_pca_result = pd.DataFrame(pca_opp_values, columns=opp_column_names)
        
        # Remover as colunas originais utilizadas no PCA
        df_reduced = df.drop(columns=team_pca_columns + opp_pca_columns)
        
        # Concatenar todos os resultados em um único DataFrame
        df = pd.concat([df_reduced.reset_index(drop=True), df_team_pca_result, df_opp_pca_result], axis=1)
        
        # Salvar o novo dataset com o PCA aplicado
    df.to_csv('pre_processamento/games_data_preproc_final.csv', index=False)
