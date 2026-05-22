import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del ejercicio
x_datos = np.array([0, 2, 4, 6, 8])
y_datos = np.array([100, 92, 85, 78, 71])
x_estimar = 5

inicio = time.time()
b, a = np.polyfit(x_datos, y_datos, 1)

# Evaluación en x = 5
y_estimado = b * x_estimar + a

y_pred = b * x_datos + a
ss_res = np.sum((y_datos - y_pred) ** 2)
ss_tot = np.sum((y_datos - np.mean(y_datos)) ** 2)
r_cuadrado = 1 - (ss_res / ss_tot)
fin = time.time()

print(f"Coeficientes obtenidos:")
print(f"Intercepto (a): {a:.4f}")
print(f"Pendiente (b): {b:.4f}")
print(f"Temperatura estimada en x = {x_estimar} cm: {y_estimado:.4f} °C")
print(f"Coeficiente R^2: {r_cuadrado:.4f}")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Generación de la gráfica
plt.figure(figsize=(8, 5))
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos originales')
plt.plot(x_datos, y_pred, color='orange', label='Línea de ajuste térmico')
plt.scatter(x_estimar, y_estimado, color='green', marker='x', s=100, zorder=6, label=f'Predicción en {x_estimar} cm')
plt.title('Regresión Lineal - Conducción Térmica en Barra')
plt.xlabel('Posición (cm)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()