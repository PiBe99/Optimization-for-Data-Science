import numpy as np
from extract_index import extract_indices_of_ones
from gradient import gradient

def min_gradient_vertex(x, P, Q_new, q_new):
    '''
    Funziona che in base alle partizioni e al gradiente calcolato in x trova il vettore che permette di
    minimizzare il gradiente. Questo permette di calcolare la descent direction
    '''
    partition = extract_indices_of_ones(P)
    grad = gradient(x, Q_new, q_new)
    #inizializzo un vettore di zeri di luneghezza pari al vettore x
    s_t = np.zeros_like(x)

    for indices in partition:
        # Controllo sulla grandezza della partizione
        if len(indices) == 0:
            continue  # Skippa se vuota
        # Ricerca dell'indice che minimizza il gradiente nella partizione
        min_index = min(indices, key=lambda i: grad[i])
        # Nell'indice corrispondente porre = 1 
        s_t[min_index] = 1
    return s_t


