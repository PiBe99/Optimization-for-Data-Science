import numpy as np
def extract_indices_of_ones(matrix):
    """
    Extracts indices where the elements are 1 for each row in the matrix.
    """
    indices_list = [list(np.where(row == 1)[0]) for row in matrix]
    return indices_list
