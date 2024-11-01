import numpy as np 
from gradient import gradient
from Data_gen import Data_generator
from min_gradient_vertex import min_gradient_vertex
from  exact_line_search import exact_line_search

def duality_gap( x, s , Q, q):
    grad = gradient(Q,x,q)
    d = s-x
    
    return np.dot(-d.T, grad)
    

