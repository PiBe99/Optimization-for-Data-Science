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