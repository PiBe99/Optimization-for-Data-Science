import numpy as np
from scipy.optimize import linprog
import time
from AllFunc import *
import gurobipy as gp
from gurobipy import GRB

if __name__ == "__main__":
    n = 10
    log_file = 'log_file_solutor.txt'
# definire i parametri da voler testare
    params_list = [
        {'K': 1, 'dim_Ker': 1, 'spectral_radius': 10, 'lambda_min': 1, 'density': 1, 'seed':111}

    
    ]
    
    for idx, params in enumerate(params_list): 
        Q, q, P, I_k = Data_generator(n, **params)

#itero su lista di parametri

        x_star, history, duality_gap, result = frank_wolfe(Q, q, P, search_methods=['exact'], **params)
                # Creazione del modello
        model = gp.Model("QP")
        with open(log_file, "a") as log:
            log.write(f'\n------- \n')
            start_time_solutor = time.strftime('%Y-%m-%d %H:%M:%S')
            log.write(f"\nEsecuzione Solutor General Purpose : {start_time_solutor}\n")
            log.write(f"Parametri:\n")

# Variabile x per solutor
            x_gurobi = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")

# Funzione obiettivo: x^T Q x + q^T x
            objective = gp.QuadExpr(0)
            for i in range(n):
                for j in range(n):
                    objective += Q[i, j] * x_gurobi[i] * x_gurobi[j]
                objective += q[i] * x_gurobi[i]

            model.setObjective(objective, GRB.MINIMIZE)

# Vincoli: somma delle variabili in ogni simplesso = 1
            for I in I_k:
                model.addConstr(gp.quicksum(x_gurobi[i] for i in I) == 1)

# Risoluzione del problema
            model.optimize()

# Stampa della soluzione ottima e scrivo sul file di log i risultati del confronto
            if model.status == GRB.OPTIMAL:
                print("Valore della funzione obiettivo (Gurobi):", model.objVal)
                log.write(f'valore solutor:{model.objVal} \n' )
                log.write(f'differenza tra solutor e FW: {history[-1] - model.objVal}\n' )
                end_time = time.strftime('%Y-%m-%d %H:%M:%S')
                log.write(f'tempo fine esecuzione Solutor: {end_time} \n' )

#controllo se la soluzione trovata dal frank wolfe rispetta i vincoli richiamando 'check_domain'

            if check_domain(x_star, P):
                log.write("\nLa soluzione soddisfa i vincoli del dominio .")
            else:
                log.write("\nLa soluzione NON soddisfa i vincoli del dominio.")


