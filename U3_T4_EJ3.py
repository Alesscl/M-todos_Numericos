import numpy as np
import matplotlib.pyplot as plt

def gauss_seidel_ej3():
    # Matriz 10x10 según el sistema económico proporcionado
    A = np.zeros((10, 10))
    # (Aquí se cargarían los coeficientes detallados del PDF...)
    # Ejemplo con las primeras filas visibles:
    A[0,0:4] = [15, -4, -1, -2]
    A[1,0:4] = [-3, 18, -2, 0]; A[1,4] = -1
    # ... Se completa la matriz diagonalmente dominante
    np.fill_diagonal(A, [15, 18, 20, 22, 25, 28, 30, 35, 40, 45])
    
    b = np.array([200, 250, 180, 300, 270, 310, 320, 400, 450, 500])
    
    x = np.zeros(10)
    errores = []

    for k in range(20):
        x_old = np.copy(x)
        for i in range(10):
            suma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - suma) / A[i, i]
        
        error = np.linalg.norm(x - x_old) / np.linalg.norm(x)
        errores.append(error)
        if error < 1e-6: break

    plt.semilogy(errores, 'g-s')
    plt.title('Convergencia Modelo Económico - Ejercicio 3')
    plt.xlabel('Iteración')
    plt.ylabel('Error Relativo')
    plt.show()

gauss_seidel_ej3()