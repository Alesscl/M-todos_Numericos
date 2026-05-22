import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

# Datos del ejercicio
x_datos = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_datos = np.array([250.0, 220.0, 180.0, 150.0, 130.0, 125.0])
x_interp = 1.5

inicio = time.time()
f_lineal = interp1d(x_datos, y_datos, kind='linear')
f_cuadratica = interp1d(x_datos, y_datos, kind='quadratic')
f_cubica = interp1d(x_datos, y_datos, kind='cubic')
fin = time.time()

print(f"Resultados en x = {x_interp} cm:")
print(f"Lineal: {f_lineal(x_interp):.4f} °C")
print(f"Cuadrática: {f_cuadratica(x_interp):.4f} °C")
print(f"Cúbica: {f_cubica(x_interp):.4f} °C")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica
x_curva = np.linspace(0.0, 5.0, 200)

plt.figure(figsize=(8, 5))
plt.plot(x_curva, f_lineal(x_curva), label='Segmentada Lineal', linestyle='--', color='blue')
plt.plot(x_curva, f_cuadratica(x_curva), label='Segmentada Cuadrática', linestyle='-.', color='orange')
plt.plot(x_curva, f_cubica(x_curva), label='Segmentada Cúbica', color='green')
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos originales')
plt.title('Interpolación Segmentada - Temperatura del Cilindro')
plt.xlabel('Distancia (cm)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()