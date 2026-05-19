import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del ejercicio
x_datos = np.array([0.5, 1.0, 1.5, 2.0])
y_datos = np.array([1.2, 2.3, 3.7, 5.2])
x_interp = 1.25

# Función de interpolación de Lagrange
def lagrange_interp(x, y, x_val):
    n = len(x)
    resultado = 0
    for i in range(n):
        termino = y[i]
        for j in range(n):
            if i != j:
                termino *= (x_val - x[j]) / (x[i] - x[j])
        resultado += termino
    return resultado

# Cálculo del tiempo y resultado
inicio = time.time()
y_interp = lagrange_interp(x_datos, y_datos, x_interp)
fin = time.time()

print(f"Deformación en x = {x_interp}: {y_interp:.4f} mm")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Generación de la gráfica
x_curva = np.linspace(0.5, 2.0, 100)
y_curva = [lagrange_interp(x_datos, y_datos, xi) for xi in x_curva]

plt.figure(figsize=(8, 5))
plt.plot(x_curva, y_curva, label='Polinomio de Lagrange', color='blue')
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos originales')
plt.scatter(x_interp, y_interp, color='green', marker='x', s=100, zorder=6, label=f'Predicción en {x_interp}')
plt.title('Interpolación de Lagrange - Esfuerzos en una Viga')
plt.xlabel('Posición (m)')
plt.ylabel('Deformación (mm)')
plt.legend()
plt.grid(True)
plt.show()