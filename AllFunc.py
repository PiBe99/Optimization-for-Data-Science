import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import orth
import matplotlib.pyplot as plt 
import time
from datetime import datetime
from scipy.sparse import random as sparse_random, triu, csr_matrix, identity
from scipy.linalg import cholesky
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
    # Set default values for optional parameters
    params = {
        'K': 10,
        'dim_Ker': 10,
        'spectral_radius': 10,
        'lambda_min': 1,
        'density': 1,
        'seed': 111
    }

    
    params.update(kwargs)

    K = params['K']
    dim_Ker = params['dim_Ker']
    spectral_radius = params['spectral_radius']
    lambda_min = params['lambda_min']
    density = params['density']
    seed = params['seed']

    # Validazione degli input
    if valida_input(K,n,dim_Ker,spectral_radius, lambda_min, density):
        print('Sto generando i dati')
    else:
        ValueError('Uno dei parametri è fuori dal range consentito')
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

def valida_input(K, n, dim_Ker, spectral_radius, lambda_min, density):
    """
    Valida i parametri di input e restituisce True se tutti i parametri sono validi,
    altrimenti restituisce False.
    """
    if not 1 <= K <= n:
        return False

    if not 0 <= dim_Ker < n:
        return False

    if dim_Ker < n and spectral_radius <= 0:
        return False

    if not 0 < lambda_min <= spectral_radius:
        return False

    if not 0 <= density <= 1:
        return False

    return True

def extract_indices_of_ones(matrix):
    """
    Estrae gli indici dove il valore della matrice P è uguale a 1

    """
    indices_list = [list(np.where(row == 1)[0]) for row in matrix]
    return indices_list

def frank_wolfe(Q, q, P, max_iter=250000, tol=1e-6, search_methods=['linear', 'exact', 'fixed'], log_file='log_file_solutor.txt', **kwargs):
    """
       Input:
        n: dimesnione del problema
        k: numero dei simplessi
        dim_Ker: numero di autovalori nulli
        spectral_radius : autovalore più grande
        lambda_min: autovalore più piccolo
        density: densità/sparsità della matrice
        seed 

    Returns:
        results: dizionario dei risultati raggruppati per metodo di ricerca del passo
        x_star: vettore x ottimo
        duality_gap: valore del duality gap in x_star
        history: valore della funzione in x_star
    """
    # Valori di defautlt 
    params = {
        'K': 10,
        'dim_Ker': 10,
        'spectral_radius': 10,
        'lambda_min': 1,
        'density': 1,
        'seed': 111
    }

    params.update(kwargs)

    K = params['K']
    dim_Ker = params['dim_Ker']
    spectral_radius = params['spectral_radius']
    lambda_min = params['lambda_min']
    density = params['density']
    seed = params['seed']


    results = {}  

    for search_method in search_methods:
        c = []  # lista per duality gap
        gamma = []  # lista per step sizes
        x_solution = []  # lista per Solutiozioni
        relative_gap = []  # lista per Relative gap
        cont = 0  # counter
        n = q.shape[0] 

        # Inizio scrittura del file log
        with open(log_file, "a") as log:
            log.write(f"\n------------------------------------------------------------")
            log.write(f"\nEsecuzione Frank-Wolfe con metodo {search_method}: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Parametri:\n")
            log.write(f"n = {n}, K = {K}, dim_Ker = {dim_Ker}, spectral_radius = {spectral_radius}, seed = {seed}\n")
            start_time = time.time()

        # Inizializzo il vettore x oer iniziare l'iterazione
        x = starting_point(P)
        history = []
        #calcolo del gradiente nel punto di inizio
        grad = gradient(x, Q, q)
        # risolvo il sottoproblema lineare trovando il vettore che minimizza il gradiente
        s = min_gradient_vertex(x, P, Q, q)
        # direzione di disciesa
        d = s - x
        # Duality gap
        duality_gap = -grad @ d

        #condizione di stop delle iterazioni
        while duality_gap > tol and cont < max_iter:
            #imposto il metodo di ricerca del passo esatto
            if search_method == 'exact':
                alpha = exact_line_search(Q, d, grad, alpha_max=1.0)
            elif search_method == 'fixed':
                alpha = 0.5
            elif search_method == 'linear':
                alpha = 2 / (cont + 2)
            else:
                raise ValueError("Metodo di ricerca del passo non riconosciuto.")

            # Aggiorno la soluzione e inizio l'iterazione t+1
            x = x + alpha * d

            grad = gradient(x, Q, q)
          
            s = min_gradient_vertex(x, P, Q, q)
            # Descent direction
            d = s - x
            # Duality gap
            duality_gap = -grad @ d

            #Aggiungo i valori nelle liste appositamente create per avere uno storico
            f_val = x.T @ Q @ x + q @ x 
            history.append(f_val)
            c.append(duality_gap)
            gamma.append(alpha)
            x_solution.append(x.copy())
            relative_gap.append(duality_gap / abs(f_val))

            #
            cont += 1
        #a fine iterazione scrivo i risultati nel file di log
        end_time = time.time()
        with open(log_file, "a") as log:
            log.write(f"\nFine esecuzione Frank-Wolfe (metodo {search_method}): {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Numero di iterazioni = {cont}\n")
            log.write(f"Tempo di esecuzione = {end_time - start_time:.4f} secondi\n")
            log.write(f"Risultati ultima iterazione:\n")
            log.write(f"Duality gap = {duality_gap}\n")
            log.write(f"Step size = {alpha}\n")
            log.write(f"Valore funzione: {f_val}\n")

        # aggiungo le liste con lo storico dei risultati nel dizionario
        results[search_method] = {
            'x_solution': x_solution,
            'duality_gap': c,
            'step_sizes': gamma,
            'iterations': np.arange(1, cont + 1),
            'val_function': history
        }

    
    return x, history, duality_gap, results

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

def gradient(x, Q, q):
    ''''
    Calcolo del gradiente se la funzione ha forma = x.T Q x + q x
    '''
    grad = 2 * Q @ x + q
    return grad

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

def plot_convergence(results):
    """
    Funzione per plottare la convergenza del duality gap per diversi metodi.
    """
    plt.figure(figsize=(14, 14))
    for method, result in results.items():
        iterations = result['iterations']
        duality_gap = result['duality_gap'] # Rimuovi il primo valore (inizializzato a zero)
        plt.plot(iterations, duality_gap, label=f"{method} stepsize", alpha = 0.85)

    plt.yscale('log')  # Usa una scala logaritmica per il duality gap
    plt.xlabel('Iterazioni')
    plt.ylabel('Duality Gap')
    #plt.title(f'Plot Convergenza \n params: {params}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.legend()
    plt.show()

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
    return dentro  

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


