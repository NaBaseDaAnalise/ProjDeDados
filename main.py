from pre_processamento.preproc_clean import full_preproc
from modelagem.classification import classification
from modelagem.handicap import regresssion
from results.results import plot_results

numero_linhas_anteriores = 5
tipo_media = 'quadratica'
search_best_params = True
# full_preproc(numero_linhas_anteriores,tipo_media)
# classification(search_best_params)
# regresssion(search_best_params, target="handicap")
# regresssion(search_best_params, target="total_points")

plot_results()