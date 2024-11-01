import numpy as np
def starting_point(P):
    '''
    Inizializzo il vettore x da cui far partire le iterazioni del frank wolfe

    il vettore x viene creato aggiungendo 1 nella  posizione in cui la
    riga corrispondente alla matrice P presenta all'indice in quesitone il primo 1
    in modo da poter soddisfare i vincoli del dominio:
    -somma di x[I_k] = 1
    -x >= 0
    '''
    k, n = P.shape
    x_start = np.zeros(n)
    for i in range(k):
        idxs = np.where(P[i] == 1)[0]
        if len(idxs) > 0:
            x_start[idxs[0]] = 1
    return x_start


