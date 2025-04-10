# import matrix_lib
# from importlib import reload
# reload(matrix_lib)

import numpy as np
import random
import timeit
from numba import jit, njit, cuda, prange
import pyfiglet

#########################

def custom_init_mat(nrows, ncols):
    return [[random.randint(0, 100) for _ in range(ncols)] for _ in range(nrows)]

def custom_init_mat_fast(nrows, ncols):
    return np.random.randint(0, 100, (nrows, ncols))

#########################

def custom_add(mat1, mat2):
    print( "custom matrix addition" )
    nrows = len(mat1[:])
    ncols = len(mat1[0])
    result = custom_init_mat_fast(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            result[i][j] = mat1[i][j] + mat2[i][j]
    return result
    # return [[ mat1[i][j] + mat2[i][j] for j in range(ncols) ] for i in range(nrows)]

def custom_mul(mat1, mat2):
    print( "custom matrix multiplication" )
    nrows_mat1 = len(mat1[:])
    nrows_mat2 = len(mat2[:])
    ncols_mat1 = len(mat1[0])
    ncols_mat2 = len(mat2[0])
    if nrows_mat1 != ncols_mat2 or nrows_mat2 != ncols_mat1:
        raise ValueError("Cannot multiply")
    result = custom_init_mat_fast(nrows_mat1, ncols_mat2)
    result = custom_mul_inside(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1)
    return result

def custom_mul_inside(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1):
    for i in range(nrows_mat1):
        for j in range(ncols_mat2):
            for k in range(ncols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result 

#########################

@njit(parallel=True)
def add_njit(mat1, mat2, range='prange'):
    print( "njit matrix addition" )
    nrows = len(mat1[:])
    ncols = len(mat1[0])
    result = custom_init_mat_fast(nrows, ncols)
    if range == 'range':
        result = mul_inside_njit_range(result, mat1, mat2, nrows, ncols)
    else:
        result = add_inside_njit_prange(result, mat1, mat2, nrows, ncols)
    return result

def add_inside_njit_range(result, mat1, mat2, nrows, ncols):
    print("range")
    for i in range(nrows):
        for j in range(ncols):
            result[i][j] += mat1[i][j] + mat2[i][j]
    return result 

@njit(parallel=True)
def add_inside_njit_prange(result, mat1, mat2, nrows, ncols):
    print("prange")
    for i in prange(nrows):
        for j in prange(ncols):
            result[i][j] += mat1[i][j] + mat2[i][j]
    return result 




def mul_njit(mat1, mat2, range='prange'):
    print( "njit matrix multiplication" )
    nrows_mat1 = len(mat1[:])
    nrows_mat2 = len(mat2[:])
    ncols_mat1 = len(mat1[0])
    ncols_mat2 = len(mat2[0])
    if nrows_mat1 != ncols_mat2 or nrows_mat2 != ncols_mat1:
        raise ValueError("Cannot multiply")
    result = custom_init_mat_fast(nrows_mat1, ncols_mat2)
    if range == 'range':
        result = mul_inside_njit_range(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1)
    else:
        result = mul_inside_njit_prange(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1)
    return result

def mul_inside_njit_range(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1):
    print("range")
    for i in range(nrows_mat1):
        for j in range(ncols_mat2):
            for k in range(ncols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result 

@njit(parallel=True)
def mul_inside_njit_prange(result, mat1, mat2, nrows_mat1, ncols_mat2, ncols_mat1):
    print("prange")
    for i in prange(nrows_mat1):
        for j in prange(ncols_mat2):
            for k in prange(ncols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result 

#########################




def print_hello():
    print("hello")
    return "bye"

if __name__ == "__main__":

    print( pyfiglet.figlet_format(" Tests ") )
    mat1 = custom_init_mat_fast(50,50)
    mat2 = custom_init_mat_fast(50,50)
    print( "mat1 =", mat1)
    print()
    print( "mat2 =", mat2)

    print()
    print( "Executing np.dot(mat1, mat2) ...")
    time_ = timeit.timeit( lambda: np.dot(mat1, mat2), number=1)
    time_ = timeit.timeit( lambda: np.dot(mat1, mat2), number=1)
    print( "it took", float(format(time_, '.2e')), "s")

    print()
    # mul_ = custom_mul(mat1, mat2)
    # print( "mat1 x mat2 =", mul_)
    print( "Executing custom_mul(mat1, mat2) ...")
    time_ = timeit.timeit( lambda: custom_mul(mat1, mat2), number=1)
    time_ = timeit.timeit( lambda: custom_mul(mat1, mat2), number=1)
    print( "it took", float(format(time_, '.2e')), "s")

    print()
    # mul_ = mul_njit(mat1, mat2, range='prange')
    # print( "mat1 x mat2 =", mul_)
    print( "Executing mul_njit(mat1, mat2, range='range') ...")
    time_ = timeit.timeit( lambda: mul_njit(mat1, mat2, range='range'), number=1)
    time_ = timeit.timeit( lambda: mul_njit(mat1, mat2, range='range'), number=1)
    print( "it took", float(format(time_, '.2e')), "s")

    print()
    # mul_ = mul_njit(mat1, mat2, range='prange')
    # print( "mat1 x mat2 =", mul_)
    print( "Executing mul_njit(mat1, mat2, range='prange') ...")
    time_ = timeit.timeit( lambda: mul_njit(mat1, mat2, range='prange'), number=1)
    time_ = timeit.timeit( lambda: mul_njit(mat1, mat2, range='prange'), number=1)
    print( "it took", float(format(time_, '.2e')), "s")



# python -c "
# import matrix_lib
# matrix_lib.print_hello()
# "
# python -c "
# print('hey')
# print('ho')
# "