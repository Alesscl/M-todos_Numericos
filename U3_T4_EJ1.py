import numpy as np
import matplotlib.pyplot as plt

def gauss_seidel_ej1():
    # Matriz de coeficientes y vector de términos independientes [cite: 28, 29, 30, 31]
    A = np.array([[10, 2, 3, 1],
                  [2, 12, 2, 3],
                  [3, 2, 15, 1],
                  [1, 3, 1, 10]], dtype=float)
    b = np.array([15, 22, 18, 10], dtype=float)
    
    x = np.zeros(len(b)) # Vector inicial de ceros
    itero, err_abs, err_rel, err_cuad = [], [], [], []

    print("Iteración | I1 | I2 | I3 | I4")
    for k in range(1, 11):
        x_old = np.copy(x)
        for i in range(len(b)):
            suma = sum(A[i][j] * x[j] for j in range(len(b)) if i != j)
            x[i] = (b[i] - suma) / A[i][i]
        
        # Cálculo de errores [cite: 14]
        e_abs = np.linalg.norm(x - x_old)
        e_rel = e_abs / np.linalg.norm(x)
        e_cuad = np.sum((x - x_old)**2)
        
        itero.append(k)
        err_abs.append(e_abs)
        err_rel.append(e_rel)
        err_cuad.append(e_cuad)
        
        print(f"{k:9d} | {x[0]:.4f} | {x[1]:.4f} | {x[2]:.4f} | {x[3]:.4f}")
        if e_rel < 1e-6: break

    # Graficación 
    plt.figure(figsize=(8,5))
    plt.plot(itero, err_abs, label='Error Absoluto')
    plt.plot(itero, err_rel, label='Error Relativo')
    plt.plot(itero, err_cuad, label='Error Cuadrático')
    plt.yscale('log')
    plt.title('Evolución de Errores - Ejercicio 1')
    plt.xlabel('Iteración')
    plt.ylabel('Error (Escala Log)')
    plt.legend()
    plt.grid(True)
    plt.show()

gauss_seidel_ej1()