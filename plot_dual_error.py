import numpy as np
import matplotlib.pyplot as plt
# Funzione per plottare il dual error
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