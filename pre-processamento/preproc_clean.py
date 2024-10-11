import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_games = pd.read_csv('../data/games_data_22-23-24.csv').copy()
df_players = pd.read_csv('../data/players_data.csv').copy()

#Remoção de valores sujos nos jogadores

cols_to_check = [
    'Team_Player1_Starters', 'Team_Player2_Starters', 'Team_Player3_Starters',
    'Team_Player4_Starters', 'Team_Player5_Starters', 'Team_Player6_Starters',
    'Team_Player7_Starters', 'Team_Player8_Starters', 'Team_Player9_Starters',
    'Team_Player10_Starters', 'Team_Player11_Starters', 'Team_Player12_Starters',
    'Team_Player13_Starters', 'Team_Player14_Starters', 'Opponent_Player1_Starters',
    'Opponent_Player2_Starters', 'Opponent_Player3_Starters', 'Opponent_Player4_Starters',
    'Opponent_Player5_Starters', 'Opponent_Player6_Starters', 'Opponent_Player7_Starters',
    'Opponent_Player8_Starters', 'Opponent_Player9_Starters', 'Opponent_Player10_Starters',
    'Opponent_Player11_Starters', 'Opponent_Player12_Starters', 'Opponent_Player13_Starters',
    'Opponent_Player14_Starters', 'Opponent_Player15_Starters', 'Opponent_Player16_Starters',
    'Team_Player15_Starters', 'Team_Player16_Starters', 'Opponent_Player17_Starters',
    'Opponent_Player18_Starters', 'Team_Player17_Starters', 'Team_Player18_Starters',
    'Team_Player19_Starters'
]

df_games[cols_to_check] = df_games[cols_to_check].replace('Team Totals', pd.NA)

#Conversão do balor da coluna Streak
def convert_streak(streak):
    if pd.isna(streak):
        return streak
    streak_type, streak_num = streak.split()
    streak_num = int(streak_num)
    if streak_type == 'W':
        return streak_num
    elif streak_type == 'L':
        return -streak_num

df_games['Streak'] = df_games['Streak'].apply(convert_streak)
df_games.head(10)
df_games.rename(columns={'Unnamed: 7': 'Resultado'}, inplace=True)
df_games.drop(columns=['G', 'Year'], inplace=True)


#Agrupamento dos jogadores

def filter_empty(players):
    return [player for player in players if pd.notna(player)]

df_games['Starters_Team'] = df_games[['Team_Player1_Starters', 'Team_Player2_Starters', 'Team_Player3_Starters',
                                      'Team_Player4_Starters', 'Team_Player5_Starters']].apply(filter_empty, axis=1)

df_games['Bench_Team'] = df_games[['Team_Player6_Starters', 'Team_Player7_Starters', 'Team_Player8_Starters',
                                   'Team_Player9_Starters', 'Team_Player10_Starters', 'Team_Player11_Starters',
                                   'Team_Player12_Starters', 'Team_Player13_Starters', 'Team_Player14_Starters',
                                   'Team_Player15_Starters', 'Team_Player16_Starters', 'Team_Player17_Starters',
                                   'Team_Player18_Starters', 'Team_Player19_Starters']].apply(filter_empty, axis=1)

df_games['Starters_Opp'] = df_games[['Opponent_Player1_Starters', 'Opponent_Player2_Starters',
                                     'Opponent_Player3_Starters', 'Opponent_Player4_Starters',
                                     'Opponent_Player5_Starters']].apply(filter_empty, axis=1)

df_games['Bench_Opp'] = df_games[['Opponent_Player6_Starters', 'Opponent_Player7_Starters', 'Opponent_Player8_Starters',
                                  'Opponent_Player9_Starters', 'Opponent_Player10_Starters', 'Opponent_Player11_Starters',
                                  'Opponent_Player12_Starters', 'Opponent_Player13_Starters', 'Opponent_Player14_Starters',
                                  'Opponent_Player15_Starters', 'Opponent_Player16_Starters', 'Opponent_Player17_Starters',
                                  'Opponent_Player18_Starters']].apply(filter_empty, axis=1)



cols_to_drop = ['Team_Player1_Starters', 'Team_Player2_Starters', 'Team_Player3_Starters',
                'Team_Player4_Starters', 'Team_Player5_Starters', 'Team_Player6_Starters',
                'Team_Player7_Starters', 'Team_Player8_Starters', 'Team_Player9_Starters',
                'Team_Player10_Starters', 'Team_Player11_Starters', 'Team_Player12_Starters',
                'Team_Player13_Starters', 'Team_Player14_Starters', 'Team_Player15_Starters',
                'Team_Player16_Starters', 'Team_Player17_Starters', 'Team_Player18_Starters',
                'Team_Player19_Starters', 'Opponent_Player1_Starters', 'Opponent_Player2_Starters',
                'Opponent_Player3_Starters', 'Opponent_Player4_Starters', 'Opponent_Player5_Starters',
                'Opponent_Player6_Starters', 'Opponent_Player7_Starters', 'Opponent_Player8_Starters',
                'Opponent_Player9_Starters', 'Opponent_Player10_Starters', 'Opponent_Player11_Starters',
                'Opponent_Player12_Starters', 'Opponent_Player13_Starters', 'Opponent_Player14_Starters',
                'Opponent_Player15_Starters', 'Opponent_Player16_Starters', 'Opponent_Player17_Starters',
                'Opponent_Player18_Starters']

df_games = df_games.drop(columns=cols_to_drop)

#Correlação
df_numeric = df_games.select_dtypes(include=[float, int])

corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# plt.show()



#Alteração da tabela para ter os dados históricos:

numero_linhas_anteriores = 3
df_games = df_games.sort_values(by='Date', ascending=False)

def calcular_media(team, date):
    
    linhas_anteriores = df_games[(df_games['Team'] == team) & (df_games['Date'] < date)].head(numero_linhas_anteriores)
    
    # if len(linhas_anteriores) > 0:
    #     print("Valores 'Team_ORB%' das 3 linhas anteriores:")
    #     for i in range(len(linhas_anteriores)):
    #         print(f"Linha {i+1}: {linhas_anteriores.iloc[i]['Team_ORB%']}")
    # else:
    #     print("Nenhuma linha anterior encontrada para este time e data.")
    
    
    medias = linhas_anteriores[current_team_columns].mean()
    # print(f"Média de 'Team_ORB%': {medias['Team_ORB%']}")

    return medias

def imputar_gmsc_players(players, row, idx):
    date = row['Date']
    team_starters_gmsc = 0  # Zera o total de GmSc da equipe
    count_players = 0  # Contador para os jogadores válidos

    # Converte a data para o formato numérico YYYYMMDD

    for player in row[players]:
        # Filtra os dados do jogador na base df_players
        player_data = df_players[df_players['Player'] == player].copy()  # Evitar problemas com SettingWithCopyWarning
        
        # Converte a coluna 'Date' de player_data para o formato numérico YYYYMMDD
        player_data.loc[:, 'Date_Num'] = pd.to_datetime(player_data['Date']).dt.strftime('%Y%m%d').astype(int)
        
        # Filtra as 3 datas anteriores mais próximas da data atual
        previous_games = player_data[player_data['Date_Num'] < date].sort_values(by='Date_Num', ascending=False).head(3)
        
        # Calcula a média de GmSc dos 3 jogos
        mean_gmsc = previous_games['GmSc'].mean()

        # Adiciona ao total se a média não for NaN
        if not pd.isna(mean_gmsc):
            team_starters_gmsc += mean_gmsc
            count_players += 1

    # Calcula a média de GmSc dos jogadores titulares da equipe
    gmsc_mean_team = team_starters_gmsc / count_players if count_players > 0 else 0

    df_games.loc[idx, f'GmSc_{players}'] = round(df_games.loc[idx, f'GmSc_{players}'] + gmsc_mean_team, 2)
    return 

team_columns = [col for col in df_games.columns if 'Team' in col and pd.api.types.is_numeric_dtype(df_games[col])]
opponent_columns = [col for col in df_games.columns if 'Opponent' in col and pd.api.types.is_numeric_dtype(df_games[col])]

for idx, row in df_games.iterrows():
    team = row['Team']
    opponent = row['Opponent']
    date = row['Date']
    
    # Preenche as colunas da equipe
    current_team_columns = team_columns
    medias = calcular_media(team, date)
    medias = [round(media, 2) for media in medias]
    df_games.loc[idx, current_team_columns] = medias

    # Preenche as colunas do oponente
    current_team_columns = opponent_columns
    medias = calcular_media(opponent, date)
    medias = [round(media, 2) for media in medias]
    df_games.loc[idx, current_team_columns] = medias
    
    # Inicializa a coluna GmSc_Starters_Team com 0.0 (float) se ainda não existir
    if 'GmSc_Starters_Team' not in df_games.columns:
        df_games['GmSc_Starters_Team'] = 0.0
        
    if 'GmSc_Starters_Opp' not in df_games.columns:
        df_games['GmSc_Starters_Opp'] = 0.0
        
    if 'GmSc_Bench_Team' not in df_games.columns:
        df_games['GmSc_Bench_Team'] = 0.0
        
    if 'GmSc_Bench_Opp' not in df_games.columns:
        df_games['GmSc_Bench_Opp'] = 0.0

    # Atualizar valores de 'W' ou 'L'
    if row['Resultado'] == 'W':
        df_games.loc[idx, 'W'] -= 1
    elif row['Resultado'] == 'L':
        df_games.loc[idx, 'L'] -= 1

    # Atualizar o valor de 'Streak' baseado no último confronto
    previous_games = df_games[(df_games['Team'] == team) & (df_games['Opponent'] == opponent) & (df_games['Date'] < date)]
    
    if not previous_games.empty:
        # Obter a data mais próxima
        last_game = previous_games.loc[previous_games['Date'].idxmax()]
        df_games.loc[idx, 'Streak'] = last_game['Streak']
    else:
        # Se não houver confronto anterior, define Streak como 0
        df_games.loc[idx, 'Streak'] = 0

    # Calcular e preencher os GameScores
    print("Getting Starter Team GameScore:")
    imputar_gmsc_players("Starters_Team", row, idx)
    
    print("Getting Starter Opponent GameScore:")
    imputar_gmsc_players("Starters_Opp", row, idx)
    
    print("Getting Bench Team GameScore:")
    imputar_gmsc_players("Bench_Team", row, idx)
    
    print("Getting Bench Opponent GameScore:")
    imputar_gmsc_players("Bench_Opp", row, idx)

new_team_column_names = {col: f"Previous_{col}" for col in team_columns}
new_opponent_column_names = {col: f"Previous_{col}" for col in opponent_columns}

df_games.rename(columns=new_team_column_names, inplace=True)
df_games.rename(columns=new_opponent_column_names, inplace=True)

pd.set_option('display.max_columns', None)

# Remoção dos nomes dos jogadores 
cols_to_drop = ['Starters_Team', 'Bench_Team', 'Starters_Opp',
                'Bench_Opp',]

df_games = df_games.drop(columns=cols_to_drop)

df_games.to_csv("games_data_preproc.csv", index=False)

print(df_games.head())