import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del ejercicio
x_datos = np.array([5, 10, 15, 20, 25])
y_datos = np.array([0.6, 1.2, 1.9, 2.5, 3.1])

# Ajuste de regresión lineal (Grado 1)
inicio = time.time()
b, a = np.polyfit(x_datos, y_datos, 1)

# Cálculo del coeficiente de determinación R^2 para el análisis de error
y_pred = b * x_datos + a
ss_res = np.sum((y_datos - y_pred) ** 2)
ss_tot = np.sum((y_datos - np.mean(y_datos)) ** 2)
r_cuadrado = 1 - (ss_res / ss_tot)
fin = time.time()

print(f"Coeficientes obtenidos:")
print(f"Intercepto (a): {a:.4f}")
print(f"Pendiente (b): {b:.4f}")
print(f"Ecuación: y = {b:.4f}x + ({a:.4f})")
print(f"Coeficiente R^2: {r_cuadrado:.4f}")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Generación de la gráfica
plt.figure(figsize=(8, 5))
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos originales')
plt.plot(x_datos, y_pred, color='blue', label=f'Línea de ajuste: y = {b:.2f}x + ({a:.2f})')
plt.title('Regresión Lineal - Resistencia de Materiales')
plt.xlabel('Carga (kN)')
plt.ylabel('Elongación (mm)')
plt.legend()
plt.grid(True)
plt.show()