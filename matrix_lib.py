# import matrix_lib
# from importlib import reload
# reload(matrix_lib)

import numpy as np
import random

def custom_init_mat(nrows, ncols):
    return [[random.randint(0, 100) for _ in range(ncols)] for _ in range(nrows)]

def custom_init_mat_fast(nrows, ncols):
    return np.random.randint(0, 100, (nrows, ncols))

def custom_add(mat1, mat2):
    print( "custom matrix addition" )
    nrows = len(mat1[:])
    ncols = len(mat1[0])
    return [[ mat1[i][j] + mat2[i][j] for j in range(ncols) ] for i in range(nrows)]

def custom_mul(mat1, mat2):
    print( "custom matrix multiplication" )
    nrows_mat1 = len(mat1[:])
    nrows_mat2 = len(mat2[:])
    ncols_mat1 = len(mat1[0])
    ncols_mat2 = len(mat2[0])
    
    # Check if multiplication is valid
    if nrows_mat1 != ncols_mat2 or nrows_mat2 != ncols_mat1:
        raise ValueError("Cannot multiply")

    # Initialize the result matrix
    result = custom_init_mat_fast(nrows_mat1, ncols_mat2)
    
    # Perform multiplication
    result = custom_mul_inside(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1)

    return result

def custom_mul_inside(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1):
    for i in range(nrows_mat1):
        for j in range(ncols_mat2):
            for k in range(ncols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result 