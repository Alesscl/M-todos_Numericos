import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros del circuito eléctrico
V = 10.0      # Voltios
R = 1000.0    # Ohmios
C = 0.001     # Faradios
RC = R * C

# dq/dt = (V - q/C) / R
def dqdt(t, q):
    return (V - q / C) / R

t0, tn = 0.0, 1.0
h = 0.05
pasos = int((tn - t0) / h)

t = np.linspace(t0, tn, pasos + 1)
q_rk4 = np.zeros(pasos + 1)
q_rk4[0] = 0.0  # Condición inicial q(0) = 0

inicio = time.time()
for i in range(pasos):
    k1 = dqdt(t[i], q_rk4[i])
    k2 = dqdt(t[i] + h/2, q_rk4[i] + (h/2)*k1)
    k3 = dqdt(t[i] + h/2, q_rk4[i] + (h/2)*k2)
    k4 = dqdt(t[i] + h, q_rk4[i] + h*k3)
    
    q_rk4[i+1] = q_rk4[i] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
fin = time.time()

# Solución analítica exacta
q_exacta = (C * V) * (1.0 - np.exp(-t / RC))
errores = np.abs(q_exacta - q_rk4)

print("Resultados numéricos en puntos clave:")
for idx in [0, 4, 8, 12, 16, 20]:
    print(f"t = {t[idx]:.2f} s | RK4 = {q_rk4[idx]:.6f} C | Exacta = {q_exacta[idx]:.6f} C | Error = {errores[idx]:.2e} C")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica de la carga q(t)
plt.figure(figsize=(8, 5))
plt.plot(t, q_exacta, label='Solución Exacta (Analítica)', color='black', linewidth=2.5)
plt.plot(t, q_rk4, '^--', label='Aproximación RK4 (h=0.05)', color='teal')
plt.title('Carga de un Capacitor en Circuito RC (RK4)')
plt.xlabel('Tiempo t (s)')
plt.ylabel('Carga q (Coulombs)')
plt.legend()
plt.grid(True)
plt.show()