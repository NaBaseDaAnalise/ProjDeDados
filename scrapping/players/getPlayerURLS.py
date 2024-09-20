import os
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
from string import ascii_lowercase
import json

from io import StringIO

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

def getAllActivePlayersForLetterURL(playersByLetterURL, saveHTMLS = False):
    # Open the player list page by letter
    driver.get(playersByLetterURL)

    if saveHTMLS:
        arq = open(f'htmls/htmlFor{playersByLetterURL[-2].upper()}.html', 'w')
        div_players = driver.find_element(By.ID, 'div_players')
        arq.write(div_players.get_attribute('outerHTML'))
        arq.close()
        time.sleep(5)

    if playersByLetterURL[-2] == 'a':
        # Wait for the "accept cookies" button to appear and click it
        try:
            # Use WebDriverWait to wait for the element to appear
            time.sleep(1)
            accept_cookies = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[class*="osano-cm-accept"]'))
            )
            accept_cookies.click()  # Click the button once found
            print("Accepted cookies.")
            time.sleep(1)  # Give it a second to process
        except Exception as e:
            print(f"Could not find or click the accept cookies button: {e}")
        
    # Wait for the table header with aria-label "To" to appear and click it twice to sort in descending order
    try:
        to_header = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//th[@aria-label="To"]'))
        )
        actions = ActionChains(driver)
        actions.click(to_header).perform()  # First click to sort ascending
        time.sleep(1)  # Wait for sorting
        actions.click(to_header).perform()  # Second click to sort descending
        time.sleep(1)  # Wait for sorting
        print('Sorted by max_year.')
    except Exception as e:
        print(f'Could not find or click the sorting button: {e}')
    
    # Collect player links for those who were active in 2024
    player_links = []
    player_rows = driver.find_elements(By.XPATH, './/table/tbody/tr')
    
    for row in player_rows:
        # Get the "To" year for each player (i.e., <td class="right" data-stat="year_max">2024</td>)
        try:
            to_year_elements = row.find_elements(By.TAG_NAME, 'td')
            to_year = to_year_elements[1].text
            print(f"Found player with To year: {to_year}")
        except Exception as e:
            print(f'Could not find row max year:\n{e}')
        
        # If the player's last year (to_year) is 2024, add their URL
        years = ["2022", "2023", "2024"]
        if to_year in years:
            # Get the player's URL
            link_element = row.find_element(By.TAG_NAME, 'th').find_element(By.TAG_NAME, 'a')
            print(link_element.get_attribute('href'))
            player_links.append({to_year: link_element.get_attribute('href')})

    return player_links

def getAllActivePlayers():
    letters = list(ascii_lowercase)
    all_player_links = {}
    for letter in letters:
        if letter == 'x':
            continue
        playersByLetterURL = f"https://www.basketball-reference.com/players/{letter}/"
        print(f"Acessando link: {playersByLetterURL}")

        # rate_limit()
        # time.sleep(0.6)

        playerLinks = getAllActivePlayersForLetterURL(playersByLetterURL)
        all_player_links[letter] = playerLinks
        
    # Save the player links into a JSON file
    with open('player_links.json', 'w') as json_file:
        json.dump(all_player_links, json_file, indent=4)

    print("Player links saved to 'player_links.json'")

def checkURLsExtraction():
    file_path = "static/player_links.json"
    needs_extraction = False

    # Check if the file exists
    if os.path.exists(file_path):
        # Check if the file is empty
        with open(file_path, "r") as file:
            content = file.read().strip()  # Remove any surrounding whitespaces
            if content == "":
                needs_extraction = True
    else:
        needs_extraction = True
    
    print(f"Needs extraction: {needs_extraction}")

    return needs_extraction
