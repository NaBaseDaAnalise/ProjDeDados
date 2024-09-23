import os
import time
import json
import pandas as pd
from selenium import webdriver
from string import ascii_lowercase
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from getPlayerURLS import getAllActivePlayers, checkURLsExtraction
from preProcessing import prepareDf

# Configura o driver do Selenium (usando o Chrome)
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # Rodar sem abrir o navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

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

# Função para retirar footer de cookies
def acceptCookies():
    driver.get('https://www.basketball-reference.com/players/a/')
    try:
        time.sleep(1)
        accept_cookies = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[class*="osano-cm-accept"]'))
        )
        accept_cookies.click()  # Click no botão de Accept All quando aparecer
        print("Accepted cookies.")
        time.sleep(1)  # Da 1 segundo para processar
    except Exception as e:
        print(f"Could not find or click the accept cookies button: {e}")

def trim_after_first_number(s):
    for i, char in enumerate(s):
        if char.isdigit():
            return s[:i]
    return s  # In case there's no digit found, return the original string

# Função para extrair todas as informações de 1 jogador na Regular Season
def getPlayerDataRegularSeason(baseURL, maxYear):
    # Cria um array dos anos em que o jogador estava ativo
    yearsToQuery = [year for year in range(2022, 2025) if year <= maxYear]
    for year in yearsToQuery[::-1]:
        # Geração do link para 1 ano
        gameLogURL = f'{baseURL}/gamelog/{year}'

        print(f'\n{100*"*"}\nLink Acessado: {gameLogURL}\n{100*"*"}\n')

        rate_limit()
        driver.get(gameLogURL)
        time.sleep(1)

        regular_season_div = False

        # Acha as divs das duas tabelas
        try:
            regular_season_div = driver.find_element(By.ID, 'all_pgl_basic')
        except Exception as e:
            print(f'Failed to locate regular season\n{e}')

        if regular_season_div != False:

            try:
                player_name_untreated = driver.find_element(By.ID, 'meta')
                player_name_untreated = player_name_untreated.find_element(By.TAG_NAME, 'h1').text
                player_name = trim_after_first_number(player_name_untreated)
            except Exception as e:
                print(e)

            # Localiza o elemento "Share & Export" e passa o mouse sobre ele para revelar o menu
            share_export_element = regular_season_div.find_element(By.XPATH, ".//span[text()='Share & Export']")
            actions = ActionChains(driver)
            actions.move_to_element(share_export_element).perform()
            print(f"Mouse em cima de share & export em all_pgl_basic [regular season]")
            time.sleep(0.3)
            
            # Localiza o botão 'Get table as CSV (for Excel)'
            csv_button = regular_season_div.find_element(By.XPATH, ".//button[@tip='Convert the table below to comma-separated values<br>suitable for use with Excel']")
            driver.execute_script("arguments[0].click();", csv_button)
            print(f"Botão 'Get table as CSV' clicado com sucesso em all_pgl_basic [regular season]")

            pre_element = driver.find_element(By.ID, "csv_pgl_basic")  # Locate the <pre> tag by ID
            csv_text = pre_element.text  # Extract the text content

            # Write the CSV content to the file
            with open("static/aux.csv", "w") as arq:
                arq.write(csv_text)

            # Remove the first 3 lines and rewrite the file
            with open("static/aux.csv", "r+") as arq:
                lines = arq.readlines()[4:]  # Skip the first 3 lines
                arq.seek(0)  # Move to the start of the file
                arq.writelines(lines)  # Overwrite with the remaining lines
                arq.truncate()  # Ensure the file size is reduced if needed
            
            try:
                df = prepareDf("static/aux.csv", player_name)
                all_data.append(df)
            except Exception as e:
                print(e)
                continue


# Função para extrair todas as informações de 1 jogador na Playoffs
def getPlayerDataPlayoffs(baseURL, maxYear):
        # Cria um array dos anos em que o jogador estava ativo
    yearsToQuery = [year for year in range(2022, 2025) if year <= maxYear]
    for year in yearsToQuery[::-1]:
        # Geração do link para 1 ano
        gameLogURL = f'{baseURL}/gamelog/{year}'

        print(f'\n{100*"*"}\nLink Acessado: {gameLogURL}\n{100*"*"}\n')

        rate_limit()
        driver.get(gameLogURL)
        time.sleep(1)

        playoffs_div = False

        try:
            playoffs_div = driver.find_element(By.ID, 'all_pgl_basic_playoffs')
        except Exception as e:
            print(f'Failed to locate playoffs\n{e}')

        if playoffs_div != False:

            try:
                player_name_untreated = driver.find_element(By.ID, 'meta')
                player_name_untreated = player_name_untreated.find_element(By.TAG_NAME, 'h1').text
                player_name = trim_after_first_number(player_name_untreated)
            except Exception as e:
                print(e)

            # Localiza o elemento "Share & Export" e passa o mouse sobre ele para revelar o menu
            share_export_element = playoffs_div.find_element(By.XPATH, ".//span[text()='Share & Export']")
            actions = ActionChains(driver)
            actions.move_to_element(share_export_element).perform()
            print(f"Mouse em cima de share & export em all_pgl_basic_playoffs [playoffs]")
            time.sleep(0.3)

            # Localiza o botão 'Get table as CSV (for Excel)'
            csv_button = playoffs_div.find_element(By.XPATH, ".//button[@tip='Convert the table below to comma-separated values<br>suitable for use with Excel']")
            driver.execute_script("arguments[0].click();", csv_button)
            print(f"Botão 'Get table as CSV' clicado com sucesso em all_pgl_basic_playoffs [playoffs]")

            pre_element = driver.find_element(By.ID, "csv_pgl_basic_playoffs")  # Locate the <pre> tag by ID
            csv_text = pre_element.text  # Extract the text content

            # Write the CSV content to the file
            with open("static/aux.csv", "w") as arq:
                arq.write(csv_text)

            # Remove the first 3 lines and rewrite the file
            with open("static/aux.csv", "r+") as arq:
                lines = arq.readlines()[4:]  # Skip the first 3 lines
                arq.seek(0)  # Move to the start of the file
                arq.writelines(lines)  # Overwrite with the remaining lines
                arq.truncate()  # Ensure the file size is reduced if needed
            try:
                df = prepareDf("static/aux.csv", player_name)
                all_data.append(df)
            except Exception as e:
                print(e)
                continue

# Function to process a single player's data for regular season and playoffs
def process_player_data(link, maxYear):
    getPlayerDataRegularSeason(link[:-5], int(maxYear))
    getPlayerDataPlayoffs(link[:-5], int(maxYear))
    allGood.append(link[:-5])
    # Update the static/allGood.txt file after processing each letter
    with open("static/allGood.txt", "w") as arq:
        arq.writelines(f"{link}\n" for link in allGood)


# Function to process all players for a given letter
def process_players_by_letter(letter):
    if letter in player_links:
        for item in player_links[letter]:
            for maxYear, link in item.items():
                if link[:-5] not in allGood:
                    process_player_data(link, maxYear)
                    combined_df = pd.concat(all_data, ignore_index=True)

                    # Append the combined DataFrame to the CSV file, write header only if the file is empty or does not exist
                    combined_df.to_csv("static/players_data.csv", index=False)

# -------------------------------------------- Bloco Principal ----------------------------------------------------- #

if checkURLsExtraction():
    getAllActivePlayers()

letters = list(ascii_lowercase)
player_links = {}

# Lista de links já processados
allGood = []

with open("static/allGood.txt", "r") as arq:
    allGood = [line.strip() for line in arq]

with open("player_links.json", "r") as linksFile:
    player_links = json.load(linksFile)

acceptCookies()

for letter in letters:
    process_players_by_letter(letter)
