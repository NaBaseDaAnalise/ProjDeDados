import time
import requests
import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains

from io import StringIO

# Configura o driver do Selenium (usando o Chrome)
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # Rodar sem abrir o navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

team_name_to_abbreviation = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHO',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}


teams_already_gone = []

teams = [
    'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS',
    'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN'
]

# Lista para armazenar todos os DataFrames
all_data = []

# Controle de requisições
request_times = []
REQUEST_LIMIT = 20  # Limite de requisições por minuto
REQUEST_INTERVAL = 60  # Intervalo de 1 minuto

# Função para controlar a taxa de requisições
def rate_limit():
    global request_times
    current_time = time.time()
    # Remove requisições antigas
    request_times = [t for t in request_times if current_time - t < REQUEST_INTERVAL]
    if len(request_times) >= REQUEST_LIMIT:
        wait_time = REQUEST_INTERVAL - (current_time - request_times[0])
        print(f"Limite de requisições atingido. Aguardando {wait_time:.2f} segundos.")
        time.sleep(wait_time)
    request_times.append(current_time)

def extract_games_from_div(driver, div_id, team, year, team_name_to_abbreviation, is_playoffs):
    
    # Localiza o elemento div
    div_element = driver.find_element(By.ID, div_id)
    
    # Localiza o elemento "Share & Export" e passa o mouse sobre ele para revelar o menu
    share_export_element = div_element.find_element(By.XPATH, ".//span[text()='Share & Export']")
    actions = ActionChains(driver)
    actions.move_to_element(share_export_element).perform()
    print(f"Mouse em cima de share & export em {div_id}")
    time.sleep(0.3)

    # Localiza o botão 'Get table as CSV (for Excel)'
    csv_button = div_element.find_element(By.XPATH, ".//button[@tip='Convert the table below to comma-separated values<br>suitable for use with Excel']")
    driver.execute_script("arguments[0].click();", csv_button)

    print(f"Botão 'Get table as CSV' clicado com sucesso em {div_id}")

    # Espera carregar o conteúdo do CSV
    csv_string = "csv_games"
    
    if is_playoffs:
        csv_string = "csv_games_playoffs"
        
    WebDriverWait(div_element, 10).until(EC.presence_of_element_located((By.ID,csv_string)))

    # Captura o conteúdo do elemento <pre id="csv_games">
    csv_element = div_element.find_element(By.ID, csv_string)
    csv_content = csv_element.text

    # Remove linhas desnecessárias no início do CSV
    csv_lines = csv_content.split('\n')
    start_index = 0
    for i, line in enumerate(csv_lines):
        if line.startswith("G,"):
            start_index = i
            break
    csv_game_data = '\n'.join(csv_lines[start_index:])

    # Adiciona as colunas "Team", "Year" e "Is_Playoffs" ao CSV
    csv_buffer = StringIO(csv_game_data)
    df = pd.read_csv(csv_buffer, header=0)  # Considera a primeira linha como header
    df["Team"] = team
    df["Year"] = year
    df["Is_Playoffs"] = is_playoffs

    # Converte a coluna 'Date' de string para um número no formato YYYYMMDD
    df['Date'] = pd.to_datetime(df['Date'], format='%a %b %d %Y')  # Converte de 'Wed Oct 25 2023' para datetime
    df['Date'] = df['Date'].dt.strftime('%Y%m%d')  # Formata para o número desejado
    df['Opponent'] = df['Opponent'].map(team_name_to_abbreviation)

    return df

def extract_box_score_data(driver, team, df, index, url, isOpponent):
    try:
        # Localiza a <div> cujo ID é "all_box-{team}-game-basic"
        div_element = driver.find_element(By.ID, f"all_box-{team}-game-basic")

        # Localiza o elemento "Share & Export" dentro da <div> encontrada
        share_export_element = div_element.find_element(By.XPATH, ".//span[text()='Share & Export']")
        actions = ActionChains(driver)
        actions.move_to_element(share_export_element).perform()
        print(f"Mouse em cima de share & export")
        time.sleep(0.3)

        # Localiza e clica no botão "Get table as CSV"
        csv_button = div_element.find_element(By.XPATH, ".//button[@tip='Convert the table below to comma-separated values<br>suitable for use with Excel']")
        driver.execute_script("arguments[0].click();", csv_button)
        print(f"Botão 'Get table as CSV' clicado com sucesso em {url}")

        # Aguarda o CSV estar disponível
        WebDriverWait(div_element, 10).until(EC.presence_of_element_located((By.ID, f"csv_box-{team}-game-basic")))

        # Captura o conteúdo do elemento CSV
        csv_element = div_element.find_element(By.ID, f"csv_box-{team}-game-basic")
        csv_content = csv_element.text

        # Remove linhas desnecessárias no início do CSV
        csv_lines = csv_content.split('\n')
        start_index = 0
        for i, line in enumerate(csv_lines):
            if line.startswith("Starters,"):
                start_index = i
                break
        csv_box_score_data = '\n'.join(csv_lines[start_index:])

        # Lê o CSV box score data
        csv_buffer = StringIO(csv_box_score_data)
        df_box = pd.read_csv(csv_buffer)

        # Para cada jogador, crie colunas exclusivas no formato Player{i}_Coluna
        for i, player_row in df_box.iterrows():
            if isOpponent:
                prefix = f"Opponent_Player{i+1}_"
            else: 
                prefix = f"Team_Player{i+1}_"
            col_name = f"{prefix}{df_box.columns[0]}"
            if col_name not in df.columns:
                df[col_name] = None  # Adiciona a coluna se não existir
            df.at[index, col_name] = player_row[df_box.columns[0]]

        print(f"Dados de box score de {team} adicionados para a linha {index}.")
    
    except Exception as e:
        print(f"Erro ao extrair dados de {team}: {e}")

def extract_four_factors(driver, team, opponent, index, df):
    
    div_element = driver.find_element(By.ID, f"all_four_factors")

    # Coleta todas as linhas da tabela de Four Factors
    rows = div_element.find_elements(By.XPATH, ".//tbody/tr")

    # Inicializa dicionários para armazenar os dados dos times
    team_data = {}
    opponent_data = {}

    # Itera sobre as linhas da tabela e coleta os dados de cada time
    for row in rows:
        # Obtém o nome do time (BOS, NYK, etc.)
        team_name_element = row.find_element(By.XPATH, ".//th[@data-stat='team_id']")
        team_name = team_name_element.text
        
        # Coleta os dados da linha (Pace, eFG%, TOV%, ORB%, FT/FGA, ORtg)
        pace = row.find_element(By.XPATH, ".//td[@data-stat='pace']").text
        efg_pct = row.find_element(By.XPATH, ".//td[@data-stat='efg_pct']").text
        tov_pct = row.find_element(By.XPATH, ".//td[@data-stat='tov_pct']").text
        orb_pct = row.find_element(By.XPATH, ".//td[@data-stat='orb_pct']").text
        ft_rate = row.find_element(By.XPATH, ".//td[@data-stat='ft_rate']").text
        off_rtg = row.find_element(By.XPATH, ".//td[@data-stat='off_rtg']").text
        
        # Verifica se o time é o team ou opponent e armazena os dados
        if team_name == team:
            team_data = {
                f"Team_Pace": pace,
                f"Team_eFG%": efg_pct,
                f"Team_TOV%": tov_pct,
                f"Team_ORB%": orb_pct,
                f"Team_FT/FGA": ft_rate,
                f"Team_ORtg": off_rtg
            }
        elif team_name == opponent:
            opponent_data = {
                f"Opponent_Pace": pace,
                f"Opponent_eFG%": efg_pct,
                f"Opponent_TOV%": tov_pct,
                f"Opponent_ORB%": orb_pct,
                f"Opponent_FT/FGA": ft_rate,
                f"Opponent_ORtg": off_rtg
            }

    # Atualiza o DataFrame com os dados do team
    for key, value in team_data.items():
        if key not in df.columns:
            df[key] = None  # Adiciona a coluna se não existir
        df.at[index, key] = value

    # Atualiza o DataFrame com os dados do opponent
    for key, value in opponent_data.items():
        if key not in df.columns:
            df[key] = None  # Adiciona a coluna se não existir
        df.at[index, key] = value

    print(f"Dados de Four Factors adicionados para {team} e {opponent} na linha {index}.")
    return df
    
# Itera pelos URLs dos times
for team in teams:
    years = [2023]
    for year in years:
        link = f"https://www.basketball-reference.com/teams/{team}/{year}_games.html"
        print(f"Acessando: {link}")  
        
        rate_limit()  # Controla a taxa de requisições
        driver.get(link)
        print(f"Navegando para a página 'Schedule & Results' em {link}")

        # Espera a página carregar
        time.sleep(0.6)
            
        try:
            all_games_df = extract_games_from_div(driver, "all_games", team, year, team_name_to_abbreviation, False)

            # Verifica se o elemento "all_games_playoffs" existe antes de tentar extrair dados dele
            try:
                all_games_playoffs_df = extract_games_from_div(driver, "all_games_playoffs", team, year, team_name_to_abbreviation, True)
            except Exception as e:
                print(f"Elemento 'all_games_playoffs' não encontrado: {e}")
                all_games_playoffs_df = pd.DataFrame()  # Cria um DataFrame vazio se o elemento não for encontrado

             # Concatena os DataFrames
            df = pd.concat([all_games_df, all_games_playoffs_df], ignore_index=True)

            
            # Lista para armazenar os índices das linhas a serem removidas
            rows_to_remove = []
            
            # Itera por cada linha do DataFrame
            for index, row in df.iterrows():
                # Construir o URL dinamicamente
                date_str = row['Date']  # O valor da data já está no formato YYYYMMDD
                opponent = row['Opponent']  # Assume que a coluna 'Opponent' já está definida no DataFrame
                opponent_index = df.columns.get_loc('Opponent')
                coluna_anterior = df.columns[opponent_index - 1]
                is_home_team = row[coluna_anterior]
                
                url = f"https://www.basketball-reference.com/boxscores/{date_str}0{team}.html"

                if is_home_team == "@":
                    url = f"https://www.basketball-reference.com/boxscores/{date_str}0{opponent}.html"

                if not (opponent in teams_already_gone):
                    try:
                        rate_limit()  # Controla a taxa de requisições
                        driver.get(url)
                        print(f"Acessando: {url}")

                        time.sleep(0.5)
                        
                        df = extract_four_factors(driver, team, opponent, index, df)
                        extract_box_score_data(driver, team, df, index, url, False)
                        extract_box_score_data(driver, opponent, df, index, url, True)
                        
                    except Exception as e:
                        print(f"Erro ao acessar {url}: {e}")
                else: 
                    rows_to_remove.append(index)
            df = df.drop(rows_to_remove).reset_index(drop=True)

        except Exception as e:
            print(f"Erro ao acessar {link}: {e}")
    
    

    all_data.append(df)
    teams_already_gone.append(team)

driver.quit()

file_path = 'games_data_copy.csv'

if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Colunas a serem removidas se existirem
    cols_to_drop = ['Notes', 'LOG', 'Attend.', 'Unnamed: 8', 'Unnamed: 5', 'Unnamed: 4', 'Unnamed: 3', 'Start (ET)']
    full_df = full_df.drop(columns=[col for col in cols_to_drop if col in full_df.columns])

    # Verifica se o arquivo já existe
    if os.path.exists(file_path):
        # Carrega o arquivo existente
        existing_df = pd.read_csv(file_path)

        # Faz o append dos novos dados no dataframe existente
        full_df = pd.concat([existing_df, full_df], ignore_index=True)
    
    # Salva os dados atualizados no arquivo CSV
    full_df.to_csv(file_path, index=False)
    
    print("Dados coletados e adicionados com sucesso.")
else:
    print("Nenhum dado foi coletado.")