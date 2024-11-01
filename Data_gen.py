import numpy as np
from scipy.linalg import orth
from scipy.sparse import random as sparse_random, triu, csr_matrix, identity
from scipy.linalg import cholesky
from valida_input import valida_input

def Data_generator(n, **kwargs):
    """
    Funzione che restituisce i seguenti Output:
    la matrice Q, il vettore q, matrice dei vincoli P e gli indici dei simplessi I_K

    input: 
    n: numerosità del problema
    K: numero dei simplessi
    dim_ker: dimensione del kernel
    lambda_min: autovalore più piccolo impostato a 1
    density: densità della matrice
    seed: seed per la generazione dei dati
    """
    # Parametri di default
    params = {
        'K': 10,
        'dim_Ker': 10,
        'spectral_radius': 10,
        'lambda_min': 1,
        'density': 1,
        'seed': 111
    }

    
    params.update(kwargs)
    #valida i dati di input
    if valida_input(K,n,dim_Ker,spectral_radius, lambda_min, density):
        print('Sto generando i dati')
    else:
        ValueError('Uno dei parametri è fuori dal range consentito')

    K = params['K']
    dim_Ker = params['dim_Ker']
    spectral_radius = params['spectral_radius']
    lambda_min = params['lambda_min']
    density = params['density']
    seed = params['seed']


    # Inizializzo il seed
    np.random.seed(seed)

    # genero Q
    if dim_Ker == n:
        rc = np.zeros(n)
    elif dim_Ker == n - 1:
        rc = [spectral_radius] + [0] * dim_Ker
    else:
        rc = [lambda_min + (spectral_radius - lambda_min) * x for x in [1] + list(np.random.rand(n - dim_Ker - 2)) + [0]]
        rc += [0] * dim_Ker

    if density == 1:
        S = np.diag(rc)
        U = orth(np.random.rand(n, n))
        Q = U @ S @ U.T  
        Q = (Q + Q.T) / 2  
    else:
  # Genera una matrice sparsa triangolare superiore
        upper = sparse_random(n, n, density=density/2, data_rvs=lambda size: np.random.choice(rc, size=size), format='csr')
        upper = triu(upper, k=1)
        # Crea una matrice simmetrica
        Q_sparse = upper + upper.T
        # Aggiungi elementi positivi al diagonale
        diag_entries = np.random.uniform(low=min(rc), high=max(rc), size=n) + n
        Q_sparse.setdiag(diag_entries)
        Q = Q_sparse.toarray()
    
    # Richiamo la funzione vincoli che genera la matrice dei vincoli P in base agli indici I_k dei simplessi
    P, indices = vincoli(n, K, seed)

    # genero vettore q
    q = np.random.rand(n)

   
    return Q, q, P, indices

def vincoli(n, k, seed):
    '''
    Funzione utilizzata in data_generator che contribuisce alla generazione dei dati
    
    Input:
    n: dimesnione del problema
    k: numero dei simplessi
    seed

    Output:
    P: matrice kxn dei vincoli
    I_k: indici delle partizioni
    '''
    #definisco una lista di numeri in base alla dimensione del problema
    indices = np.arange(n)
    np.random.seed(seed)
    #utilizzo shuffle per randomizzare l'ordine
    np.random.shuffle(indices)
    #splitto il vettore in k simplessi
    I_k = np.array_split(indices, k)

    P = np.zeros((k, n))
    for j in range(k):    
        pos = I_k[j]
        P[j][pos] = 1
    return P, I_k

