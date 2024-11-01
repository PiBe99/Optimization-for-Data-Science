import numpy as np

def exact_line_search(Q, d, grad, alpha_max=1.0):
    ''''
    Questa funzione utilizza il calcolo del gradiente e della direzione servendosi anche di Q e q
    per calcolare il passo esatto per calcolare successivamente la nuova x.
    '''
    denom = 2 * d.T @ Q @ d
    if denom == 0:
        return alpha_max
    alpha = -grad @ d / denom
    return min(alpha_max, max(0, alpha))






