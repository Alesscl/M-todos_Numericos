import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del ejercicio
x_datos = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
y_datos = np.array([0.32, 0.30, 0.28, 0.27, 0.26, 0.25])
x_interp = 35.0

def diferencias_divididas(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def evaluar_newton(coef, x_datos, x_val):
    n = len(coef)
    resultado = coef[0]
    producto = 1.0
    for i in range(1, n):
        producto *= (x_val - x_datos[i-1])
        resultado += coef[i] * producto
    return resultado

inicio = time.time()
coeficientes = diferencias_divididas(x_datos, y_datos)
y_interp = evaluar_newton(coeficientes, x_datos, x_interp)
fin = time.time()

print(f"Coeficientes del polinomio: {coeficientes}")
print(f"Coeficiente de arrastre en V = {x_interp} m/s: {y_interp:.4f}")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica
x_curva = np.linspace(10, 60, 100)
y_curva = [evaluar_newton(coeficientes, x_datos, xi) for xi in x_curva]

plt.figure(figsize=(8, 5))
plt.plot(x_curva, y_curva, label='Polinomio de Newton', color='purple')
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos reales')
plt.scatter(x_interp, y_interp, color='green', marker='x', s=100, zorder=6, label=f'Predicción en {x_interp} m/s')
plt.title('Interpolación de Newton - Coeficiente de Arrastre')
plt.xlabel('Velocidad (m/s)')
plt.ylabel('Coeficiente de Arrastre (Cd)')
plt.legend()
plt.grid(True)
plt.show()