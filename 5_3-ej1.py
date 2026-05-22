import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

# Datos del ejercicio
x_datos = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_datos = np.array([0.0, -1.5, -2.8, -3.0, -2.7, -2.0])
x_interp = 2.5

# Medición de tiempos de procesamiento e interpolación
inicio = time.time()
f_lineal = interp1d(x_datos, y_datos, kind='linear')
f_cuadratica = interp1d(x_datos, y_datos, kind='quadratic')
f_cubica = interp1d(x_datos, y_datos, kind='cubic')
fin = time.time()

print(f"Resultados en x = {x_interp} m:")
print(f"Lineal: {f_lineal(x_interp):.4f} mm")
print(f"Cuadrática: {f_cuadratica(x_interp):.4f} mm")
print(f"Cúbica: {f_cubica(x_interp):.4f} mm")
print(f"Tiempo de procesamiento total: {(fin - inicio) * 1000:.4f} ms")

# Gráfica comparativa
x_curva = np.linspace(0.0, 5.0, 200)

plt.figure(figsize=(8, 5))
plt.plot(x_curva, f_lineal(x_curva), label='Segmentada Lineal', linestyle='--', color='blue')
plt.plot(x_curva, f_cuadratica(x_curva), label='Segmentada Cuadrática', linestyle='-.', color='orange')
plt.plot(x_curva, f_cubica(x_curva), label='Segmentada Cúbica (Spline)', color='green')
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos originales')
plt.title('Interpolación Segmentada - Deflexión en Viga')
plt.xlabel('Longitud (m)')
plt.ylabel('Deflexión (mm)')
plt.legend()
plt.grid(True)
plt.show()