import random
import numpy as np
from numba import jit, njit, vectorize, cuda


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

class Matrix_Selma:
    def __init__(self, values):
        # D'abord, vérifier le type de 'values' avant de l'assigner à self.values
        if not isinstance(values, np.ndarray):
            raise TypeError("Type error: the matrix must be a NumPy array.")
        
        # Ensuite, assigner 'values' à l'attribut 'self.values'
        self.values = values

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
