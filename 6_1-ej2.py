import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del problema mecánico
g = 9.81   # m/s^2
m = 2.0    # kg
k = 0.5    # kg/s

t0, tn = 0.0, 10.0
n = 50
h = (tn - t0) / n

t = np.linspace(t0, tn, n + 1)
v_euler = np.zeros(n + 1)
v_euler[0] = 0.0  # Condición inicial v(0) = 0

inicio = time.time()
for i in range(n):
    # dv/dt = g - (k/m)*v
    f_derivada = g - (k / m) * v_euler[i]
    v_euler[i+1] = v_euler[i] + h * f_derivada
fin = time.time()

# Solución analítica exacta
v_analitico = ((m * g) / k) * (1.0 - np.exp(-(k / m) * t))
errores = np.abs(v_analitico - v_euler)

print("Resultados de pasos seleccionados:")
for idx in [0, 10, 20, 30, 40, 50]:
    print(f"t = {t[idx]:.2f} s | Euler = {v_euler[idx]:.4f} m/s | Analítico = {v_analitico[idx]:.4f} m/s | Error = {errores[idx]:.4f} m/s")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(t, v_analitico, label='Solución Exacta (Analítica)', color='black', linewidth=2)
plt.plot(t, v_euler, 's--', label='Aproximación de Euler (n=50)', color='orange')
plt.title('Método de Euler - Velocidad en Caída Libre con Resistencia')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.legend()
plt.grid(True)
plt.show()