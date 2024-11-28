

def calculoDeRisco(odd): 
    risco = (1/odd) #penalizar mais por odds maiores #risco sendo uma combinação de métricas
    return risco

def selecionarOdd(casaDeAposta1, casaDeAposta2, casaDeAposta3):
    odd = max(casaDeAposta1, casaDeAposta2, casaDeAposta3) #pegar a odd mais favorável 
    return odd

def intervaloVale (intervalo):
    if intervalo: #entre positivo e negativo 
        return 
        
def calculoDeRetornoEsperado(acuracia, risco, valor_maximo_aposta=100): 
    retorno = (acuracia) - risco
    
    if retorno <= 0:
        print("Não vale apostar")
    else: 
        print(f"Vale apostar. Retorno esperado: {retorno:.2f}")
        
        # Diferenciação para níveis de risco
        if risco < 0.3:
            print("Nível de risco baixo.")
        elif risco < 0.6:
            print("Nível de risco médio.")
        else:
            print("Nível de risco alto.")

        # Sugerir um valor de aposta com base no risco
        valor_sugerido = sugestaoDeValorAposta(risco, valor_maximo_aposta)
        print(f"Valor sugerido para apostar: {valor_sugerido:.2f}")


def sugestaoDeValorAposta(risco, max_valor=50):
    # Sugerir valores de aposta com base no riscos
    if risco <= 0.2:
        valor_sugerido = max_valor
    elif risco <= 0.4:
        valor_sugerido = max_valor * 0.75
    elif risco <= 0.6:
        valor_sugerido = max_valor * 0.5
    else:
        valor_sugerido = max_valor * 0.25

    return valor_sugerido