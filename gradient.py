def gradient(x, Q, q):
    ''''
    Calcolo del gradiente se la funzione ha forma = x.T Q x + q x
    '''
    grad = 2 * Q @ x + q
    return grad





