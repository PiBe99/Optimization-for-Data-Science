import numpy as np

def check_domain(x, P):
    """
    Funzione che viene utilizzata per avere riscontro sul rispetto delle condizioni e vincoli del dominio.
    -somma di x[I_k] = 1
    -x >= 0

    Parametri:
        first_point : punto di partenza da verificare verificare
        P : ogni riga rappresenta un vincolo
    
    Restituisce:
        dentro :  True se e solo se x soddisfa tutti i K vincoli
    """
    x = np.asarray(x).reshape(-1)  
    K, _ = P.shape

    #CONRTOLLO RISPETTO VINCOLI
    dentro = np.allclose(P @ x, np.ones(K), atol=1e-6) and np.all(x >= 0)