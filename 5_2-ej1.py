import numpy as np
import matplotlib.pyplot as plt
import time

# Datos del ejercicio
x_datos = np.array([50.0, 100.0, 150.0, 200.0])
y_datos = np.array([0.12, 0.35, 0.65, 1.05])
x_interp = 125.0

# Función para calcular la tabla de diferencias divididas de Newton
def diferencias_divididas(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:,0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
            
    return coef[0, :] # Retorna los coeficientes c0, c1, c2...

# Función para evaluar el polinomio de Newton
def evaluar_newton(coef, x_datos, x_val):
    n = len(coef)
    resultado = coef[0]
    producto = 1.0
    for i in range(1, n):
        producto *= (x_val - x_datos[i-1])
        resultado += coef[i] * producto
    return resultado

# Cálculo del tiempo y resultado
inicio = time.time()
coeficientes = diferencias_divididas(x_datos, y_datos)
y_interp = evaluar_newton(coeficientes, x_datos, x_interp)
fin = time.time()

print(f"Coeficientes del polinomio: {coeficientes}")
print(f"Deformación en F = {x_interp} N: {y_interp:.4f} mm")
print(f"Tiempo de procesamiento: {(fin - inicio) * 1000:.4f} ms")

# Gráfica
x_curva = np.linspace(50, 200, 100)
y_curva = [evaluar_newton(coeficientes, x_datos, xi) for xi in x_curva]

plt.figure(figsize=(8, 5))
plt.plot(x_curva, y_curva, label='Polinomio de Newton', color='blue')
plt.scatter(x_datos, y_datos, color='red', zorder=5, label='Datos originales')
plt.scatter(x_interp, y_interp, color='green', marker='x', s=100, zorder=6, label=f'Predicción en {x_interp} N')
plt.title('Interpolación de Newton - Deformación de Material')
plt.xlabel('Carga F (N)')
plt.ylabel('Deformación (mm)')
plt.legend()
plt.grid(True)
plt.show()