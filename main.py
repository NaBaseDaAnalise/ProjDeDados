from pre_processamento.preproc_clean import full_preproc
from modelagem.classification import classification

numero_linhas_anteriores = 5
search_best_params = False
# full_preproc(numero_linhas_anteriores)
classification(search_best_params)