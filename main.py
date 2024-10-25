from pre_processamento.preproc_clean import full_preproc
from modelagem.classification import classification
from modelagem.total_points import pred_total_points
from modelagem.handicap import pred_handicap

numero_linhas_anteriores = 5
tipo_media = 'quadratica'
search_best_params = False
# full_preproc(numero_linhas_anteriores,tipo_media)
# classification(search_best_params)
# pred_total_points(search_best_params)
pred_handicap(search_best_params)