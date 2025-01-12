import numpy as np

def multiply_matrices_basic(A, B):
    if A.shape[1] == B.shape[0]:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
        for row in range(A.shape[0]):
            for col in range(B.shape[1]):
                for i in range(A.shape[1]):
                    C[row, col] += A[row, i] * B[i, col]
        return C
    else:
        return "Matrix Multiplication is not possible"


def multiply_matrices_parallel(A, B):
    return np.dot(A, B)


