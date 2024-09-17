import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/games_data.csv')


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

df[cols_to_check] = df[cols_to_check].replace('Team Totals', pd.NA)

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

df['Streak'] = df['Streak'].apply(convert_streak)
df.head(10)
df.rename(columns={'Unnamed: 7': 'Resultado'}, inplace=True)
df.drop(columns=['G', 'Year'], inplace=True)


#Agrupamento dos jogadores

df['Starters_Team'] = df[['Team_Player1_Starters', 'Team_Player2_Starters', 'Team_Player3_Starters',
                          'Team_Player4_Starters', 'Team_Player5_Starters']].values.tolist()

df['Bench_Team'] = df[['Team_Player6_Starters', 'Team_Player7_Starters', 'Team_Player8_Starters',
                       'Team_Player9_Starters', 'Team_Player10_Starters', 'Team_Player11_Starters',
                       'Team_Player12_Starters', 'Team_Player13_Starters', 'Team_Player14_Starters',
                       'Team_Player15_Starters', 'Team_Player16_Starters', 'Team_Player17_Starters',
                       'Team_Player18_Starters', 'Team_Player19_Starters']].values.tolist()

df['Starters_Opp'] = df[['Opponent_Player1_Starters', 'Opponent_Player2_Starters',
                         'Opponent_Player3_Starters', 'Opponent_Player4_Starters',
                         'Opponent_Player5_Starters']].values.tolist()

df['Bench_Opp'] = df[['Opponent_Player6_Starters', 'Opponent_Player7_Starters', 'Opponent_Player8_Starters',
                      'Opponent_Player9_Starters', 'Opponent_Player10_Starters', 'Opponent_Player11_Starters',
                      'Opponent_Player12_Starters', 'Opponent_Player13_Starters', 'Opponent_Player14_Starters',
                      'Opponent_Player15_Starters', 'Opponent_Player16_Starters', 'Opponent_Player17_Starters',
                      'Opponent_Player18_Starters']].values.tolist()


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

df = df.drop(columns=cols_to_drop)

#Correlação
df_numeric = df.select_dtypes(include=[float, int])

corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# plt.show()


#Alteração da tabela para ter os dados históricos:

numero_linhas_anteriores = 3
df = df.sort_values(by='Date', ascending=False)

def calcular_media(team, date):
    
    linhas_anteriores = df[(df['Team'] == team) & (df['Date'] < date)].head(numero_linhas_anteriores)
    
    # if len(linhas_anteriores) > 0:
    #     print("Valores 'Team_ORB%' das 3 linhas anteriores:")
    #     for i in range(len(linhas_anteriores)):
    #         print(f"Linha {i+1}: {linhas_anteriores.iloc[i]['Team_ORB%']}")
    # else:
    #     print("Nenhuma linha anterior encontrada para este time e data.")
    
    
    medias = linhas_anteriores[current_team_columns].mean()
    # print(f"Média de 'Team_ORB%': {medias['Team_ORB%']}")

    return medias

team_columns = [col for col in df.columns if 'Team' in col and pd.api.types.is_numeric_dtype(df[col])]
opponent_columns = [col for col in df.columns if 'Opponent' in col and pd.api.types.is_numeric_dtype(df[col])]

for idx, row in df.iterrows():
    team = row['Team']
    opponent = row['Opponent']
    date = row['Date']
    
    current_team_columns = team_columns
    medias = calcular_media(team, date)
    df.loc[idx, current_team_columns] = medias

    
    current_team_columns = opponent_columns
    medias = calcular_media(opponent, date)
    df.loc[idx, current_team_columns] = medias


new_team_column_names = {col: f"Previous_{col}" for col in team_columns}
new_opponent_column_names = {col: f"Previous_{col}" for col in opponent_columns}

df.rename(columns=new_team_column_names, inplace=True)
df.rename(columns=new_opponent_column_names, inplace=True)


pd.set_option('display.max_columns', None)

print(df.head())