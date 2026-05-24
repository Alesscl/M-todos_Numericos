import numpy as np
import matplotlib.pyplot as plt
import time

# Sistema vectorizado: Y = [y1, y2] -> dY/dt = [y2, -2*y2 - 5*y1]
def sistema_derivadas(t, Y):
    y1, y2 = Y[0], Y[1]
    dy1_dt = y2
    dy2_dt = -2.0 * y2 - 5.0 * y1
    return np.array([dy1_dt, dy2_dt])

t0, tn = 0.0, 5.0
h = 0.1
pasos = int((tn - t0) / h)

t = np.linspace(t0, tn, pasos + 1)
# Matriz de estados: fila 0 es y1 (posición), fila 1 es y2 (velocidad)
Y = np.zeros((2, pasos + 1))
Y[:, 0] = [1.0, 0.0]  # Condiciones iniciales y1(0)=1, y2(0)=0

inicio = time.time()
for i in range(pasos):
    state = Y[:, i]
    k1 = sistema_derivadas(t[i], state)
    k2 = sistema_derivadas(t[i] + h/2, state + (h/2)*k1)
    k3 = sistema_derivadas(t[i] + h/2, state + (h/2)*k2)
    k4 = sistema_derivadas(t[i] + h, state + h*k3)
    
    Y[:, i+1] = state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
fin = time.time()

print("Resultados de la trayectoria de la masa:")
for idx in [0, 10, 20, 30, 40, 50]:
    print(f"t = {t[idx]:.1f} s | Posición (y1) = {Y[0, idx]:.4f} | Velocidad (y2) = {Y[1, idx]:.4f}")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica de la trayectoria
plt.figure(figsize=(8, 5))
plt.plot(t, Y[0, :], label='Posición $y_1(t)$ (Masa)', color='blue', linewidth=2)
plt.plot(t, Y[1, :], label='Velocidad $y_2(t)$', color='orange', linestyle='--')
plt.axhline(0, color='black', linestyle=':', alpha=0.5)
plt.title('Dinámica de un Resorte Amortiguado (RK4)')
plt.xlabel('Tiempo t (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()