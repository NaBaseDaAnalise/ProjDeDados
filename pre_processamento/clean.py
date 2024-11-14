import pandas as pd

def filter_empty(players):
    return [player for player in players if pd.notna(player)]
    
#Conversão do valor da coluna Streak
def convert_streak(streak):
    if pd.isna(streak):
        return streak
    streak_type, streak_num = streak.split()
    streak_num = int(streak_num)
    if streak_type == 'W':
        return streak_num
    elif streak_type == 'L':
        return -streak_num
    
def clean(df_games):
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
    
    df_games.rename(columns={'Unnamed: 7': 'Resultado'}, inplace=True)

        
    #Agrupamento dos jogadores
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

    df_games.drop(columns=cols_to_drop, inplace=True)
    df_games['Streak'] = df_games['Streak'].apply(convert_streak)

def execute_clean_games(): 
    print("Limpando dataset de games\n")

    df_games = pd.read_csv('data/games_data_22-23-24.csv').copy()

    clean(df_games)
    
    df_games.to_csv('data/cleaned_games_data_22-23-24.csv', index=False)