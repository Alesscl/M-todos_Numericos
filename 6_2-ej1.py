import numpy as np
import matplotlib.pyplot as plt
import time

# Definición de la EDO: dT/dx = -0.25 * (T - 25)
def dydx(x, T):
    return -0.25 * (T - 25.0)

# Condiciones iniciales y configuración
x0, xn = 0.0, 2.0
T0 = 100.0
h = 0.1
pasos = int((xn - x0) / h)

# Inicialización de arreglos
x = np.linspace(x0, xn, pasos + 1)
T_rk4 = np.zeros(pasos + 1)
T_rk4[0] = T0

# Algoritmo de Runge-Kutta de 4to Orden (RK4)
inicio = time.time()
for i in range(pasos):
    k1 = dydx(x[i], T_rk4[i])
    k2 = dydx(x[i] + h/2, T_rk4[i] + (h/2)*k1)
    k3 = dydx(x[i] + h/2, T_rk4[i] + (h/2)*k2)
    k4 = dydx(x[i] + h, T_rk4[i] + h*k3)
    
    T_rk4[i+1] = T_rk4[i] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
fin = time.time()

# Solución exacta para la comparación
T_exacta = 25.0 + 75.0 * np.exp(-0.25 * x)
errores = np.abs(T_exacta - T_rk4)

print("Resultados numéricos en puntos clave:")
for idx in [0, 5, 10, 15, 20]:
    print(f"x = {x[idx]:.1f} m | RK4 = {T_rk4[idx]:.4f} °C | Exacta = {T_exacta[idx]:.4f} °C | Error = {errores[idx]:.6f} °C")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica del perfil térmico
plt.figure(figsize=(8, 5))
plt.plot(x, T_exacta, label='Solución Exacta (Analítica)', color='black', linewidth=2.5)
plt.plot(x, T_rk4, 'o', label='Aproximación RK4 (h=0.1)', color='crimson')
plt.title('Perfil de Temperatura a lo largo del Tubo (RK4)')
plt.xlabel('Distancia x (m)')
plt.ylabel('Temperatura T (°C)')
plt.legend()
plt.grid(True)
plt.show()