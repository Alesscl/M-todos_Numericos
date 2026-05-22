import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del ejercicio
x_datos = np.array([50, 70, 90, 110, 130])
y_datos = np.array([15, 21, 27, 33, 39])
x_predecir = 100

inicio = time.time()
b, a = np.polyfit(x_datos, y_datos, 1)

# Evaluación en x = 100
y_predicho = b * x_predecir + a

y_pred = b * x_datos + a
ss_res = np.sum((y_datos - y_pred) ** 2)
ss_tot = np.sum((y_datos - np.mean(y_datos)) ** 2)
r_cuadrado = 1 - (ss_res / ss_tot)
fin = time.time()

print(f"Coeficientes obtenidos:")
print(f"Intercepto (a): {a:.4f}")
print(f"Pendiente (b): {b:.4f}")
print(f"Caudal predicho a {x_predecir} kPa: {y_predicho:.4f} L/min")
print(f"Coeficiente R^2: {r_cuadrado:.4f}")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Generación de la gráfica
plt.figure(figsize=(8, 5))
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos reales')
plt.plot(x_datos, y_pred, color='purple', label='Línea de ajuste hidráulico')
plt.scatter(x_predecir, y_predicho, color='green', marker='x', s=100, zorder=6, label=f'Predicción a {x_predecir} kPa')
plt.title('Regresión Lineal - Caudal vs Presión')
plt.xlabel('Presión (kPa)')
plt.ylabel('Caudal (L/min)')
plt.legend()
plt.grid(True)
plt.show()