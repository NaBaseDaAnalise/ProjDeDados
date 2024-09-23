import pandas as pd

# Function to convert "MM:SS" format to total minutes as a float with error handling
def convert_minutes_played(time_str):
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return round(minutes + seconds / 60.0, 2)
    except Exception as e:
        print(f"Error converting value: {time_str} - {e}")
        return 0  # Return 0 if there's any error in the conversion

def prepareDf(csvFileName, playerName):
    # Assuming your DataFrame is already processed up to this point
    df = pd.read_csv(csvFileName)
    df = df.drop(columns='Rk')

    df.columns.values[0] = "Game"
    df.columns.values[4] = "Home"
    df.columns.values[6] = "Result/Diff"

    # Convert the 'GS' column to numeric, setting errors='coerce' to turn non-numeric values into NaN
    df['GS'] = pd.to_numeric(df['GS'], errors='coerce')

    # Drop rows where 'GS' is NaN (i.e., non-numeric values)
    df = df.dropna(subset=['GS'])    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Home'] = df['Home'].map({"@": 1}).fillna(0).astype(int)

    # Get the index position of the 'Result/Diff' column
    result_diff_index = df.columns.get_loc('Result/Diff')

    # Split the 'Result/Diff' column
    df['Win'] = df['Result/Diff'].str.extract(r'([WL])')  # Extract 'W' or 'L'
    df['Diff'] = df['Result/Diff'].str.extract(r'([+-]?\d+)')  # Extract the number inside parentheses

    # Convert 'Win' to 1 for 'W' and 0 for 'L'
    df['Win'] = df['Win'].map({'W': 1, 'L': 0}).astype(int)

    # Convert 'Diff' to int
    df['Diff'] = df['Diff'].astype(int)

    # Drop the original 'Result/Diff' column
    df = df.drop(columns=['Result/Diff'])

    # Insert 'Win' and 'Diff' columns into the same position as 'Result/Diff'
    df.insert(result_diff_index, 'Win', df.pop('Win'))
    df.insert(result_diff_index + 1, 'Diff', df.pop('Diff'))

    df['Age'] = df['Age'].str.split('-').str[0].astype(int)

    # Apply the function to the 'MP' column and replace the original values
    df['MP'] = df['MP'].apply(convert_minutes_played)

    columns_to_fill = [
        'GS', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'ORB', 'TRB', 'DRB', 
        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-', 'FG%', '3P%', 'FT%', 'GmSc'
    ]

    df[columns_to_fill] = df[columns_to_fill].fillna(0)

    df = df.astype({
        'GS': 'int',
        'FG': 'int',
        'FGA': 'int',
        '3P': 'int',
        '3PA': 'int',
        'FT': 'int',
        'FTA': 'int',
        'ORB': 'int',
        'TRB': 'int',
        'DRB': 'int',
        'AST': 'int',
        'STL': 'int',
        'BLK': 'int',
        'TOV': 'int',
        'PF': 'int',
        'PTS': 'int',
        '+/-': 'int',
        'FG%': 'float',
        '3P%': 'float',
        'FT%': 'float',
        'GmSc': 'float',
    })

    df.insert(0, "Player", playerName.strip())

    return df
