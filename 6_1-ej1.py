import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros del circuito
R = 1000       # Ohmios
C = 0.001      # Faradios
V_fuente = 5   # Voltios
RC = R * C

# Configuración del método de Euler
t0, tn = 0.0, 5.0
n = 20
h = (tn - t0) / n

# Inicialización de arreglos
t = np.linspace(t0, tn, n + 1)
V_euler = np.zeros(n + 1)
V_euler[0] = 0.0  # Condición inicial V(0) = 0

# Algoritmo de Euler y medición de tiempo
inicio = time.time()
for i in range(n):
    # dV/dt = (1/RC) * (V_fuente - V)
    f_derivada = (1.0 / RC) * (V_fuente - V_euler[i])
    V_euler[i+1] = V_euler[i] + h * f_derivada
fin = time.time()

# Solución analítica exacta
V_analitico = V_fuente * (1.0 - np.exp(-t / RC))
errores = np.abs(V_analitico - V_euler)

# Despliegue de resultados numéricos clave
print("Resultados de pasos seleccionados:")
for idx in [0, 4, 8, 12, 16, 20]:
    print(f"t = {t[idx]:.2f} s | Euler = {V_euler[idx]:.4f} V | Analítico = {V_analitico[idx]:.4f} V | Error = {errores[idx]:.4f} V")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Generación de la gráfica
plt.figure(figsize=(8, 5))
plt.plot(t, V_analitico, label='Solución Analítica (Exacta)', color='black', linewidth=2)
plt.plot(t, V_euler, 'o--', label='Aproximación de Euler (n=20)', color='blue')
plt.title('Método de Euler - Carga de un Capacitor')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje V(t)')
plt.legend()
plt.grid(True)
plt.show()