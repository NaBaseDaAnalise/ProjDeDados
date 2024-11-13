from pre_processamento.clean import execute_clean_games
from pre_processamento.preproc_merge import full_preproc
from pre_processamento.preproc_pca import apply_pca_team_stats
from modelagem.classification import classification
from modelagem.handicap import regresssion
from results.results import plot_results
from itertools import product

# Ex: Número de linhas anteriores, normalização de colunas, PCA do players_data,
# PCA do games_pre_proc / só das estatísticas dos times, ao invés de utilizar a média dos jogos anteriores, 
# utilizar uma média ponderada (com pesos diferentes).

# Valores que você quer testar para cada variável inicial
linhas_anteriores_values = [3, 5, 7]
tipo_media_values = ['simples','linear', 'quadratica']
pca_players_values = [True, False]
pca_team_stats_values = [0, 1, 2]
search_best_params = True

param_combinations = product(
    linhas_anteriores_values,
    tipo_media_values,
    pca_players_values,
)

execute_clean_games()

for numero_linhas_anteriores, tipo_media, pca_players in param_combinations:
    print(f"Iniciando experimento com: numero_linhas_anteriores={numero_linhas_anteriores}, "
        f"tipo_media={tipo_media}, "
        f"pca_players={pca_players}, pca_team_stats={pca_team_stats_values}\n")
    
    print("Aplicando pré-processamento inicial")
    
    full_preproc(numero_linhas_anteriores, tipo_media, pca_players)
    
    for pca_team_stats in pca_team_stats_values:
        
        print(f"Experimentando com: numero_linhas_anteriores={numero_linhas_anteriores}, "
            f"tipo_media={tipo_media}, "
            f"pca_players={pca_players}, pca_team_stats={pca_team_stats}")

        apply_pca_team_stats(pca_team_stats)
        
        classification(numero_linhas_anteriores, tipo_media, pca_players, pca_team_stats, search_best_params)
        regresssion(search_best_params, target="handicap")
        regresssion(search_best_params, target="total_points")

        # plot_results()  # Descomentar se deseja visualizar os resultados para cada experimento