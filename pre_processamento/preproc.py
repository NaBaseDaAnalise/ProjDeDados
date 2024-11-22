import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import ast
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class PreprocMerge:
    def __init__(self, numero_linhas_anteriores, tipo_media, pca_players=True, use_previous_preproc=True):
        self.df_games = pd.read_csv('data/cleaned_games_data_22-23-24.csv').copy()
        self.df_players = pd.read_csv('data/players_data.csv').copy()
        self.numero_linhas_anteriores = numero_linhas_anteriores
        self.tipo_media = tipo_media
        self.pca_players = pca_players
        self.use_previous_preproc = use_previous_preproc

        self.pca_columns = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 
                            'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-']
        
        # Calcular os pesos do PCA apenas uma vez
        self.pca = PCA(n_components=1)
        self.pca.fit(self.df_players[self.pca_columns])
        self.pca_weights = self.pca.components_[0]

    def calcular_media(self, previous_values):
        """
        Calcula a média de GmSc com diferentes tipos de ponderação para várias colunas.
        
        Parâmetros:
        - previous_values: DataFrame contendo os jogos anteriores.
        - tipo_media: 'simples' para média simples, 'linear' para ponderação linear, 'quadratica' para ponderação quadrática.
        
        Retorna:
        - Uma lista com a média de GmSc para cada coluna, com base no tipo de ponderação escolhido.
        """
        if isinstance(previous_values, pd.Series):
            # Se for uma Série, tratar como uma única coluna
            previous_values = previous_values.to_frame() 
            
        medias = []  # Lista para armazenar as médias

        # Iterar sobre cada coluna do DataFrame
        for coluna in previous_values.columns:
            valores = previous_values[coluna].values  # Extrair os valores da coluna
            n = len(valores)  # Número de jogos disponíveis
            
            # Caso seja média simples
            if self.tipo_media == 'simples':
                media = sum(valores) / n if n > 0 else 0
                medias.append(media)  # Adicionar a média à lista

            # Caso seja média ponderada linearmente
            elif self.tipo_media == 'linear':
                media_ponderada_linear = 0
                soma_pesos = 0
                for i in range(n):
                    peso = i + 1  # Peso é o índice + 1
                    media_ponderada_linear += valores[i] * peso
                    soma_pesos += peso
                media = media_ponderada_linear / soma_pesos if soma_pesos > 0 else 0
                medias.append(media)

            # Caso seja média ponderada quadraticamente
            elif self.tipo_media == 'quadratica':
                media_ponderada_quadratica = 0
                soma_pesos = 0
                for i in range(n):
                    peso = (i + 1) ** 2  # Peso é o quadrado do índice + 1
                    media_ponderada_quadratica += valores[i] * peso
                    soma_pesos += peso
                media = media_ponderada_quadratica / soma_pesos if soma_pesos > 0 else 0
                medias.append(media)

            # Caso o tipo de média não seja reconhecido
            else:
                raise ValueError("Tipo de média desconhecido. Use 'simples', 'linear' ou 'quadratica'.")

        return medias  # Retornar a lista de médias
    
    
    def imputar_gmsc_players(self, players, row, idx):
        date = row['Date']
        team_players_score = 0  # Zera o total de GmSc da equipe
        count_players = 0  # Contador para os jogadores válidos
        
        for player in ast.literal_eval(row[players]):
            # Filtra os dados do jogador na base df_players
            player_data = self.df_players[self.df_players['Player'] == player].copy()
            
            # Filtra as datas anteriores mais próximas da data atual
            previous_games = player_data[player_data['Date_Num'] < date].head(self.numero_linhas_anteriores)

            if self.pca_players:
                # Passo 2: Se tiver múltiplas linhas, aplica PCA normalmente
                if len(previous_games[self.pca_columns]) > 1:
                    pca_values = self.pca.transform(previous_games[self.pca_columns])
                    mean_pca = pca_values.mean()  # ou qualquer outra métrica desejada
                
                elif len(previous_games[self.pca_columns]) == 1:
                    # Passo 3: Se tiver apenas uma linha, use os pesos pré-calculados para fazer uma projeção
                    single_line = previous_games[self.pca_columns].iloc[0].values
                    mean_pca = np.dot(single_line, self.pca_weights)  # Projeção da linha pelos pesos pré-calculados       

                # Verifica se o resultado não é NaN e atualiza a pontuação da equipe
                if  len(previous_games[self.pca_columns]) != 0:
                    team_players_score += mean_pca
                    count_players += 1
                
            else: 
                        
                # Calcula a média de GmSc dos ultimos jogos
                mean_gmsc = self.calcular_media(previous_games['GmSc'])[0]
                
                # Adiciona ao total se a média não for NaN
                if not pd.isna(mean_gmsc):
                    team_players_score += mean_gmsc
                    count_players += 1

        # Calcula a média de GmSc dos jogadores titulares da equipe
        gmsc_mean_team = team_players_score / count_players if count_players > 0 else 0

        self.df_games.loc[idx, f'Score_{players}'] = round(self.df_games.loc[idx, f'Score_{players}'] + gmsc_mean_team, 2)
        
    def previous_preproc(self):
        self.df_games.sort_values(by='Date', ascending=False)

        team_columns = [col for col in self.df_games.columns if 'Team' in col and pd.api.types.is_numeric_dtype(self.df_games[col])]
        opponent_columns = [col for col in self.df_games.columns if 'Opponent' in col and pd.api.types.is_numeric_dtype(self.df_games[col])]
        
        # Converte a coluna 'Date' de df_players para o formato numérico YYYYMMDD e coloca em ordem de data
        self.df_players.loc[:, 'Date_Num'] = pd.to_datetime(self.df_players['Date']).dt.strftime('%Y%m%d').astype(int)
        self.df_players.sort_values(by='Date_Num', ascending=False)
        
        for idx, row in self.df_games.iterrows():
            team = row['Team']
            opponent = row['Opponent']
            date = row['Date']
            
            # Preenche as colunas das estatísticas equipe
            current_team_columns = team_columns
            linhas_anteriores = self.df_games[(self.df_games['Team'] == team) & (self.df_games['Date'] < date)].head(self.numero_linhas_anteriores)
            medias = self.calcular_media(linhas_anteriores[current_team_columns])
            medias = [round(media, 2) for media in medias]
            self.df_games.loc[idx, current_team_columns] = medias

            # Preenche as colunas das estatísticas do oponente
            current_team_columns = opponent_columns
            linhas_anteriores = self.df_games[(self.df_games['Team'] == team) & (self.df_games['Date'] < date)].head(self.numero_linhas_anteriores)
            medias = self.calcular_media(linhas_anteriores[current_team_columns])
            medias = [round(media, 2) for media in medias]
            self.df_games.loc[idx, current_team_columns] = medias

            # Atualizar valores de 'W' ou 'L'
            if row['Resultado'] == 'W':
                self.df_games.loc[idx, 'W'] -= 1
            elif row['Resultado'] == 'L':
                self.df_games.loc[idx, 'L'] += 1

            # Filtrar os jogos anteriores
            previous_games = self.df_games[(self.df_games['Team'] == team) & (self.df_games['Opponent'] == opponent) & (self.df_games['Date'] < date)]

            if not previous_games.empty:
                # Obter o jogo mais recente
                last_game = previous_games.loc[previous_games['Date'].idxmax()]
                
                # Atualizar o 'Streak' baseado no último confronto
                self.df_games.loc[idx, 'Previous_Streak'] = last_game['Streak']
                
                # Selecionar os últimos 3 jogos
                last_games = previous_games.nlargest(self.numero_linhas_anteriores, 'Date')
                
                # Calcular a média de 'Tm' e 'Opp' dos últimos 3 jogos
                tm_mean = self.calcular_media(last_games['Tm'])
                tm_mean = round(tm_mean[0], 1)
                opp_mean = self.calcular_media(last_games['Opp'])
                opp_mean = round(opp_mean[0], 1)
                
                # Atualizar os valores no DataFrame
                self.df_games.loc[idx, 'Previous_Tm'] = tm_mean
                self.df_games.loc[idx, 'Previous_Opp'] = opp_mean
            else:
                # Se não houver confronto anterior, define 'Streak' como 0
                self.df_games.loc[idx, 'Previous_Streak'] = 0
                self.df_games.loc[idx, 'Previous_Tm'] = 0
                self.df_games.loc[idx, 'Previous_Opp'] = 0

            # Inicializa a coluna GmSc_Starters_Team com 0.0 (float) se ainda não existir
            if 'Score_Starters_Team' not in self.df_games.columns:
                self.df_games['Score_Starters_Team'] = 0.0
            if 'Score_Starters_Opp' not in self.df_games.columns:
                self.df_games['Score_Starters_Opp'] = 0.0
            if 'Score_Bench_Team' not in self.df_games.columns:
                self.df_games['Score_Bench_Team'] = 0.0
            if 'Score_Bench_Opp' not in self.df_games.columns:
                self.df_games['Score_Bench_Opp'] = 0.0
                
            # Calcular e preencher os GameScores
            print("Getting Starter Team GameScore:")
            self.imputar_gmsc_players("Starters_Team", row, idx)
            
            print("Getting Starter Opponent GameScore:")
            self.imputar_gmsc_players("Starters_Opp", row, idx)
            
            print("Getting Bench Team GameScore:")
            self.imputar_gmsc_players("Bench_Team", row, idx)
            
            print("Getting Bench Opponent GameScore:")
            self.imputar_gmsc_players("Bench_Opp", row, idx)

        new_team_column_names = {col: f"Previous_{col}" for col in team_columns}
        new_opponent_column_names = {col: f"Previous_{col}" for col in opponent_columns}

        self.df_games.rename(columns=new_team_column_names, inplace=True)
        self.df_games.rename(columns=new_opponent_column_names, inplace=True)
        
    def full_preproc(self):
        file_path = f"pre_processamento/preproc_datasets/experiment:{self.numero_linhas_anteriores}{self.tipo_media}{self.pca_players}.csv"

        if os.path.exists(file_path) and self.use_previous_preproc:
            print("Preproc file already exists - using existing one\n")
        else:
            print("Starting preproc\n")
            self.previous_preproc()

            # Remoção dos nomes dos jogadores, Data e colunas atreladas ao resultado e rename
            cols_to_drop = ['Starters_Team', 'Bench_Team', 'Starters_Opp',
                            'Bench_Opp','Streak','Date','G', 'Year']
            self.df_games = self.df_games.drop(columns=cols_to_drop)

            # Analisar e preencher valores faltantes apenas nas colunas numéricas
            numeric_cols = self.df_games.select_dtypes(include=['float64', 'int64']).columns
            self.df_games[numeric_cols] = self.df_games[numeric_cols].fillna(self.df_games[numeric_cols].mean())

            # Converter as colunas categóricas 'Team' e 'Opponent' para valores numéricos
            label_encoder = LabelEncoder()
            self.df_games['Team'] = label_encoder.fit_transform(self.df_games['Team'])
            self.df_games['Opponent'] = label_encoder.fit_transform(self.df_games['Opponent'])

            self.df_games.to_csv(f"pre_processamento/preproc_datasets/experiment:{self.numero_linhas_anteriores}{self.tipo_media}{self.pca_players}.csv", index=False)

            #Correlação
            df_numeric = self.df_games.select_dtypes(include=[float, int])
            corr_matrix = df_numeric.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

            # plt.show()
        
            print(self.df_games.head())
        
    def apply_pca_team_stats(self, n_components):

        df = pd.read_csv(f"pre_processamento/preproc_datasets/experiment:{self.numero_linhas_anteriores}{self.tipo_media}{self.pca_players}.csv").copy()

        if n_components != 0:

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
            
        # Salvar o novo dataset com o PCA aplicado ou não, no formato a ser utilizado depois
        df.to_csv('pre_processamento/games_data_preproc.csv', index=False)
