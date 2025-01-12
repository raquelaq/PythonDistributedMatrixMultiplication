import hazelcast
import numpy as np
from multiprocessing import Pool

def split_matrix(matrix, num_parts):
    """Divide una matriz en partes iguales."""
    rows = matrix.shape[0]
    part_size = rows // num_parts
    return [matrix[i * part_size: (i + 1) * part_size, :] for i in range(num_parts)]

def multiply_submatrices(sub_matrix_A, matrix_B):
    """Multiplica una submatriz por una matriz completa."""
    return np.dot(sub_matrix_A, matrix_B)

def process_submatrix(index, map_name, matrix_B):
    """Procesa una submatriz almacenada en el mapa de Hazelcast."""
    client = hazelcast.HazelcastClient()  # Cada proceso necesita su propio cliente
    distributed_map = client.get_map(map_name).blocking()
    sub_matrix_A = np.array(distributed_map.get(f"part_{index}"))
    result = multiply_submatrices(sub_matrix_A, matrix_B).tolist()
    client.shutdown()
    return result

def distributed_matrix_multiplication(matrix_A, matrix_B, num_workers):
    """Realiza la multiplicación distribuida de matrices usando Hazelcast."""
    client = hazelcast.HazelcastClient()

    map_name = "matrix_parts"
    distributed_map = client.get_map(map_name).blocking()

    sub_matrices_A = split_matrix(matrix_A, num_workers)

    for i, sub_matrix in enumerate(sub_matrices_A):
        distributed_map.put(f"part_{i}", sub_matrix.tolist())

    with Pool(num_workers) as pool:
        results = pool.starmap(process_submatrix, [(i, map_name, matrix_B) for i in range(num_workers)])

    result_matrix = np.vstack(results)

    client.shutdown()

    return result_matrix

# Ejemplo de uso
if __name__ == "__main__":
    # Dimensiones de las matrices
    rows_A, cols_A, cols_B = 8, 4, 6

    # Genera matrices aleatorias
    matrix_A = np.random.randint(1, 10, (rows_A, cols_A))
    matrix_B = np.random.randint(1, 10, (cols_A, cols_B))

    print("Matrix A:")
    print(matrix_A)

    print("Matrix B:")
    print(matrix_B)

    # Multiplicación distribuida
    num_workers = 4
    result = distributed_matrix_multiplication(matrix_A, matrix_B, num_workers)

    print("Resulting Matrix:")
    print(result)
