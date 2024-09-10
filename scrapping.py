import time
import requests
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

teams = [
    'ATL'
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

# Itera pelos URLs dos times
for team in teams:
    years = [2024]
    for year in years:
        link = f"https://www.basketball-reference.com/teams/{team}/{year}_games.html"
        print(f"Acessando: {link}")  
        
        rate_limit()  # Controla a taxa de requisições
        driver.get(link)
        print(f"Navegando para a página 'Schedule & Results' em {link}")

        # Espera a página carregar
        time.sleep(0.6)
            
        try:
            # Localiza o elemento "Share & Export" e passa o mouse sobre ele para revelar o menu
            share_export_element = driver.find_element(By.XPATH, "//span[text()='Share & Export']")
            actions = ActionChains(driver)
            actions.move_to_element(share_export_element).perform()
            print(f"Mouse em cima de share & export")
            time.sleep(0.3)

            # Localizando o botão 'Get table as CSV (for Excel)' pelo atributo 'tip'
            csv_button = driver.find_element(By.XPATH, "//button[@tip='Convert the table below to comma-separated values<br>suitable for use with Excel']")
            driver.execute_script("arguments[0].click();", csv_button)

            print(f"Botão 'Get table as CSV' clicado com sucesso em {link}")

            # Espera carregar o conteúdo do CSV
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "csv_games")))

            # Captura o conteúdo do elemento <pre id="csv_games">
            csv_element = driver.find_element(By.ID, "csv_games")
            csv_content = csv_element.text
            
            # Remove linhas desnecessárias no início do CSV
            csv_lines = csv_content.split('\n')
            start_index = 0
            for i, line in enumerate(csv_lines):
                if line.startswith("G,"):
                    start_index = i
                    break
            csv_game_data = '\n'.join(csv_lines[start_index:])
            
            # Adiciona as colunas "Team" e "Year" ao CSV
            csv_buffer = StringIO(csv_game_data)  # Utiliza csv_data em vez de csv_content
            df = pd.read_csv(csv_buffer, header=0)  # Considera a primeira linha como header
            df["Team"] = team
            df["Year"] = year

            # Converte a coluna 'Date' de string para um número no formato YYYYMMDD
            df['Date'] = pd.to_datetime(df['Date'], format='%a %b %d %Y')  # Converte de 'Wed Oct 25 2023' para datetime
            df['Date'] = df['Date'].dt.strftime('%Y%m%d')  # Formata para o número desejado
            df['Opponent'] = df['Opponent'].map(team_name_to_abbreviation)

            # Itera por cada linha do DataFrame
            for index, row in df.iterrows():
                # Construir o URL dinamicamente
                date_str = row['Date']  # O valor da data já está no formato YYYYMMDD
                opponent = row['Opponent']  # Assume que a coluna 'Opponent' já está definida no DataFrame

                url = f"https://www.basketball-reference.com/boxscores/{date_str}0{team}.html"

                print(f"Acessando: {url}")
                
                try:
                    rate_limit()  # Controla a taxa de requisições
                    driver.get(url)
                    time.sleep(0.3)
                    if "404" in driver.title:
                        # Se a página 404 for encontrada, tenta acessar a página com a sigla do adversário
                        url = f"https://www.basketball-reference.com/boxscores/{date_str}0{opponent}.html"
                        print(f"Página não encontrada para {team}. Tentando: {url}")
                        
                        rate_limit()
                        driver.get(url)
                        time.sleep(0.3)  

                    # Verifica se a página foi carregada corretamente
                    if "404" in driver.title:
                        print(f"Página não encontrada para {team} e {opponent}. Pulando...")
                        continue
                    
                    # Localiza a <div> cujo ID é "all_box-{team}-game-basic"
                    div_element = driver.find_element(By.ID, f"all_box-{team}-game-basic")

                    # Localiza o elemento "Share & Export" dentro da <div> encontrada
                    share_export_element = div_element.find_element(By.XPATH, ".//span[text()='Share & Export']")
                    actions = ActionChains(driver)
                    actions.move_to_element(share_export_element).perform()
                    print(f"Mouse em cima de share & export")
                    time.sleep(0.3)

                    # Localiza e clica no botão "Get table as CSV" dentro da <div> encontrada
                    csv_button = div_element.find_element(By.XPATH, ".//button[@tip='Convert the table below to comma-separated values<br>suitable for use with Excel']")
                    driver.execute_script("arguments[0].click();", csv_button)
                    
                    print(f"Botão 'Get table as CSV' clicado com sucesso em {url}")

                    WebDriverWait(div_element, 10).until(EC.presence_of_element_located((By.ID, f"csv_box-{team}-game-basic")))

                    # Captura o conteúdo do elemento <pre id="csv_games">
                    csv_element = div_element.find_element(By.ID, f"csv_box-{team}-game-basic")
                    csv_content = csv_element.text
                    
                    # Remove linhas desnecessárias no início do CSV
                    csv_lines = csv_content.split('\n')
                    start_index = 0
                    for i, line in enumerate(csv_lines):
                        if line.startswith("Starters,"):
                            start_index = i
                            break
                    csv_game_data = '\n'.join(csv_lines[start_index:])

                    # Adiciona as colunas "Team" e "Year" ao CSV
                    csv_buffer = StringIO(csv_game_data)
                    df_game = pd.read_csv(csv_buffer, header=0)
                    df_game["Team"] = team
                    df_game["Year"] = year

                    # Adiciona o DataFrame à lista
                    all_data.append(df_game)

                except Exception as e:
                    print(f"Erro ao acessar {url}: {e}")

        except Exception as e:
            print(f"Erro ao acessar {link}: {e}")

# Fecha o driver do Selenium
driver.quit()

# Combina todos os DataFrames em um único DataFrame
if all_data:
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv('games_data.csv', index=False)
    print("Dados coletados e salvos com sucesso.")
else:
    print("Nenhum dado foi coletado.")
