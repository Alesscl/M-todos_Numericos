import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros térmicos
T0 = 90.0
T_amb = 25.0
k = 0.07

t0, tn = 0.0, 30.0
n = 30
h = (tn - t0) / n

t = np.linspace(t0, tn, n + 1)
T_euler = np.zeros(n + 1)
T_euler[0] = T0  # Condición inicial T(0) = 90

inicio = time.time()
for i in range(n):
    # dT/dt = -k * (T - T_amb)
    f_derivada = -k * (T_euler[i] - T_amb)
    T_euler[i+1] = T_euler[i] + h * f_derivada
fin = time.time()

# Solución analítica exacta
T_analitico = T_amb + (T0 - T_amb) * np.exp(-k * t)
errores = np.abs(T_analitico - T_euler)

print("Resultados de pasos seleccionados:")
for idx in [0, 6, 12, 18, 24, 30]:
    print(f"t = {t[idx]:.2f} min | Euler = {T_euler[idx]:.4f} °C | Analítico = {T_analitico[idx]:.4f} °C | Error = {errores[idx]:.4f} °C")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica
plt.figure(figsize=(8, 5))
plt.plot(t, T_analitico, label='Solución Exacta (Analítica)', color='black', linewidth=2)
plt.plot(t, T_euler, 'v--', label='Aproximación de Euler (n=30)', color='purple')
plt.title('Método de Euler - Enfriamiento Térmico de Newton')
plt.xlabel('Tiempo (minutos)')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()