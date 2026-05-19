import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del ejercicio
x_datos = np.array([1.0, 2.5, 4.0, 5.5])
y_datos = np.array([85, 78, 69, 60])
x_interp = 3.0

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

inicio = time.time()
y_interp = lagrange_interp(x_datos, y_datos, x_interp)
fin = time.time()

print(f"Temperatura en x = {x_interp}: {y_interp:.4f} °C")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Generación de la gráfica
x_curva = np.linspace(1.0, 5.5, 100)
y_curva = [lagrange_interp(x_datos, y_datos, xi) for xi in x_curva]

plt.figure(figsize=(8, 5))
plt.plot(x_curva, y_curva, label='Polinomio de Lagrange', color='orange')
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos originales')
plt.scatter(x_interp, y_interp, color='green', marker='x', s=100, zorder=6, label=f'Predicción en {x_interp}')
plt.title('Interpolación de Lagrange - Temperatura del Motor')
plt.xlabel('Profundidad (cm)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()