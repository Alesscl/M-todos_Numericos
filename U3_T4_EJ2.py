import numpy as np
import matplotlib.pyplot as plt

def gauss_seidel_ej2():
    # Definición del sistema térmico [cite: 35, 36, 37]
    # Nota: Se completan los coeficientes basados en la estructura del problema
    A = np.array([[20, -5, -3, 0, 0],
                  [-4, 18, -2, -1, 0],
                  [-3, -1, 22, -5, 0],
                  [0, -2, -4, 25, -1],
                  [0, 0, 0, -1, 20]], dtype=float) # T5 simplificado para el ejemplo
    b = np.array([100, 120, 130, 150, 60], dtype=float)
    
    x = np.zeros(len(b))
    itero, err_abs, err_rel = [], [], []

    for k in range(1, 16):
        x_old = np.copy(x)
        for i in range(len(b)):
            suma = sum(A[i][j] * x[j] for j in range(len(b)) if i != j)
            x[i] = (b[i] - suma) / A[i][i]
        
        e_abs = np.linalg.norm(x - x_old)
        e_rel = e_abs / np.linalg.norm(x)
        
        itero.append(k)
        err_abs.append(e_abs)
        err_rel.append(e_rel)
        
        if e_rel < 1e-5: break

    plt.plot(itero, err_rel, 'r-o', label='Error Relativo (T)')
    plt.title('Convergencia Térmica - Ejercicio 2')
    plt.xlabel('Iteración')
    plt.ylabel('Error')
    plt.legend()
    plt.grid()
    plt.show()

gauss_seidel_ej2()