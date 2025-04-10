import random
import numpy as np
from numba import jit, njit, vectorize, cuda

#ADDITION
@jit(nopython=True)   
def add_jit_(mat1, mat2):
    return mat1 + mat2

@njit(parallel=True)
def add_njit_(mat1, mat2):
    return mat1 + mat2

@vectorize(['float64(float64, float64)'])
def add_vectorize_(mat1, mat2):
    return mat1 + mat2

@cuda.jit
def add_cuda_kernel(mat1, mat2, result):
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        result[i, j] = mat1[i, j] + mat2[i, j]


# MULTIPLICATION
def mult_matrices_naive(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        raise ValueError("Matrices cannot be multiplied: the number of columns of mat1 must equal the number of rows of mat2.")
    
    mult_mat = []
    for i in range(len(mat1)):
        row_mult = []
        for j in range(len(mat2[0])):
            val = 0
            for k in range(len(mat2)):
                val += mat1[i][k] * mat2[k][j]
            row_mult.append(val)
        mult_mat.append(row_mult)
    
    return mult_mat

def mult_matrices_numpy(mat1, mat2):
    return np.dot(mat1, mat2)

@jit(nopython=True)
def mult_matrices_jit(mat1, mat2):
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("Multiplication not possible.")
    
    result = np.zeros((mat1.shape[0], mat2.shape[1]), dtype=np.float64)

    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[1]):
            for k in range(mat1.shape[1]):
                result[i, j] += mat1[i, k] * mat2[k, j]
    
    return result

@njit(parallel=True)
def mult_matrices_njit(mat1, mat2):
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("Multiplication not possible.")
    
    result = np.zeros((mat1.shape[0], mat2.shape[1]), dtype=np.float64)

    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[1]):
            for k in range(mat1.shape[1]):
                result[i, j] += mat1[i, k] * mat2[k, j]
    
    return result

# @vectorize(['float64(float64, float64)'])
# def mult_matrices_vectorize(a, b):
#     return a * b

@cuda.jit
def mult_matrices_cuda(mat1, mat2, result):
    row, col = cuda.grid(2)
    if row < mat1.shape[0] and col < mat2.shape[1]:
        temp = 0
        for k in range(mat1.shape[1]):
            temp += mat1[row, k] * mat2[k, col]
        result[row, col] = temp

def mult_matrices_cuda_wrapper(mat1, mat2):
    n, m = mat1.shape
    m2, p = mat2.shape

    result = np.zeros((n, p), dtype=np.float64)

    mat1_device = cuda.to_device(mat1)
    mat2_device = cuda.to_device(mat2)
    result_device = cuda.to_device(result)

    threads_per_block = (16, 16)
    blocks_per_grid = (int(np.ceil(n / threads_per_block[0])), int(np.ceil(p / threads_per_block[1])))

    mult_matrices_cuda[blocks_per_grid, threads_per_block](mat1_device, mat2_device, result_device)
    result_device.copy_to_host(result)

    return result



class Matrix_Selma:
    def __init__(self, values):
        # D'abord, vérifier le type de 'values' avant de l'assigner à self.values
        if not isinstance(values, np.ndarray):
            raise TypeError("Type error: the matrix must be a NumPy array.")
        
        # Ensuite, assigner 'values' à l'attribut 'self.values'
        self.values = values
#########Addtion###########################
    def __add__(self, mat2):
        return np.add(self.values, mat2.values)

    def add_numpy(self, mat2):
        return np.add(self.values, mat2.values)

    def add_jit(self, mat2):
        return add_jit_(self.values, mat2.values)
    
    def add_njit(self, mat2):
        return add_njit_(self.values, mat2.values)
    
    def add_vectorize(self, mat2):
        return add_vectorize_(self.values, mat2.values)
    
    def add_cuda(self, mat2):
        n, m = self.values.shape
        result = np.zeros((n, m), dtype=np.float64)

        # Initialisation des matrices sur le GPU
        mat1_device = cuda.to_device(self.values)
        mat2_device = cuda.to_device(mat2.values)
        result_device = cuda.to_device(result)

        # Spécification du nombre de threads et de blocs pour CUDA
        threads_per_block = (16, 16)
        blocks_per_grid = (int(np.ceil(n / threads_per_block[0])), int(np.ceil(m / threads_per_block[1])))

        # Appel du kernel CUDA
        add_cuda_kernel[blocks_per_grid, threads_per_block](mat1_device, mat2_device, result_device)

        # Copier le résultat du GPU vers l'hôte (CPU)
        result_device.copy_to_host(result)

        return result
    
#######MULTIPLICATION#############################

    def mult_naive(self, mat2):
        return mult_matrices_naive(self.values, mat2.values)

    def mult_numpy(self, mat2):
        return np.dot(self.values, mat2.values)

    def mult_jit(self, mat2):
        return mult_matrices_jit(self.values, mat2.values)

    def mult_njit(self, mat2):
        return mult_matrices_njit(self.values, mat2.values)

    # def mult_vectorize(self, mat2):
    #     return mult_matrices_vectorize(self.values, mat2.values)

    def mult_cuda(self, mat2):
        return mult_matrices_cuda_wrapper(self.values, mat2.values)
    

#---------------------------------------------------------------------------------------------------------
mat1 = np.random.rand(3, 3)
mat2 = np.random.rand(3, 3)

M1 = Matrix_Selma(mat1)
M2 = Matrix_Selma(mat2)

print("Addition Naïve:")
print(M1 + M2)

print("\nAddition avec NumPy:")
print(M1.add_numpy(M2))

print("\nAddition avec JIT:")
print(M1.add_jit(M2)) 
print("\nAddition avec NJIT:")
print(M1.add_njit(M2))  

print("\nAddition vectorisée:")
print(M1.add_vectorize(M2))  

print("\nAddition avec CUDA (GPU):")
print(M1.add_cuda(M2))  


print("Multiplication Naïve:")
print(M1.mult_naive(M2))

print("\nMultiplication avec NumPy:")
print(M1.mult_numpy(M2))

print("\nMultiplication avec JIT:")
print(M1.mult_jit(M2))  

print("\nMultiplication avec NJIT:")
print(M1.mult_njit(M2))

# print("\nMultiplication vectorisée:")
# print(M1.mult_vectorize(M2))

print("\nMultiplication avec CUDA (GPU):")
print(M1.mult_cuda(M2))
