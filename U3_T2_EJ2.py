import numpy as np
import matplotlib.pyplot as plt

def gauss_jordan_pivot_determinante(A, b):
    """
    Resuelve un sistema de ecuaciones Ax = b mediante el método de Gauss-Jordan con pivoteo parcial
    e imprime el determinante de A para verificar si el sistema tiene solución única.

    Además, si el sistema es determinado (determinante distinto de 0), se genera una gráfica del
    error del residuo ||Ax - b|| en función de las iteraciones (una por cada pivote principal).
    """
    n = len(A)
    # Matriz aumentada
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    
    # Cálculo del determinante de A
    det_A = np.linalg.det(A)
    
    # Verificar si el sistema es determinado o indeterminado
    if np.isclose(det_A, 0):
        mensaje = f"Determinante de A: {det_A:.5f}. El sistema es indeterminado o no tiene solución única."
        print(mensaje)
        return None
    
    mensaje = f"Determinante de A: {det_A:.5f}. El sistema tiene solución única."
    print(mensaje)
    
    # Historial de errores del residuo por iteración (||Ax - b||_2)
    errores = []
    
    # Aplicación del método de Gauss-Jordan con pivoteo
    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        # Normalización de la fila pivote
        Ab[i] = Ab[i] / Ab[i, i]

        # Eliminación en otras filas
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[j, i] * Ab[i]

        # Cálculo del error del residuo con la solución aproximada actual
        x_aprox = Ab[:, -1]
        residuo = A @ x_aprox - b
        error_iter = np.linalg.norm(residuo, ord=2)
        errores.append(error_iter)

    # Extraer la solución
    x = Ab[:, -1]

    # Graficar errores si corresponde
    if len(errores) > 0:
        plt.figure()
        plt.plot(range(1, n + 1), errores, marker='o')
        plt.yscale('log')
        plt.title('Evolución del error del residuo')
        plt.xlabel('Iteración')
        plt.ylabel('||Ax - b||₂ (escala log)')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return x

# Definir el sistema de ecuaciones
A_test = np.array([
    [ 3, -2,  5, -1,  4,  2, -3,  1,  2],
    [-2,  4, -3,  1,  5, -1,  2, -4,  3],
    [ 5, -1,  2, -3,  4,  1, -2,  3, -1],
    [ 1, -3,  4, -2,  5, -1,  2, -1,  4],
    [ 2,  3, -1,  4, -2,  5, -3,  1, -2],
    [-3,  2,  4, -1,  3, -2,  5, -1,  1],
    [ 4, -1,  3,  2, -3,  1, -2,  5, -4],
    [-1,  5, -2,  3,  4, -1,  2, -3,  1],
    [ 3, -2,  5, -1,  4,  2, -3,  1, -5]
], dtype=float)

b_test = np.array([-8, 7, -6, 5, 12, -9, 10, 3, -2], dtype=float)

# Resolver el sistema
solucion_test = gauss_jordan_pivot_determinante(A_test, b_test)

# Imprimir la solución si existe
if solucion_test is not None:
    print("Solución del sistema:", solucion_test)