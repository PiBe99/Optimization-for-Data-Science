from first_point import starting_point
from min_gradient_vertex import min_gradient_vertex
from gradient import gradient
from exact_line_search import exact_line_search
import time
import numpy as np



def frank_wolfe(Q, q, P, max_iter=150000, tol=1e-6, search_methods=['linear', 'exact', 'fixed'], log_file='log_file_solutor.txt', **kwargs):
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