import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función del Ejercicio 3
def f(x):
    return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección con tabla y errores
def biseccion(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        print("El método de bisección no es aplicable en el intervalo dado.")
        return None
    
    iteraciones = []
    c_old = None

    for i in range(max_iter):
        c = (a + b) / 2
        f_c = func(c)
        
        if c_old is None:
            err_abs = None
            err_rel = None
            err_cuad = None
        else:
            err_abs = abs(c - c_old)
            err_rel = abs((c - c_old) / c) if c != 0 else None
            err_cuad = err_abs**2

        iteraciones.append([i+1, a, b, c, f_c, err_abs, err_rel, err_cuad])

        if abs(f_c) < tol or (err_abs is not None and err_abs < tol):
            break

        if func(a) * f_c < 0:
            b = c
        else:
            a = c
        c_old = c

    tabla = pd.DataFrame(iteraciones, columns=[
        "Iteración","a","b","c","f(c)","Error absoluto","Error relativo","Error cuadrático"
    ])
    return tabla

# Puntos de interpolación (4 puntos en [0,1])
x_points = np.array([0.0, 0.25, 0.5, 1.0])
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(0, 1, 200)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Bisección sobre el polinomio interpolante
tabla = biseccion(lambda x: lagrange_interpolation(x, x_points, y_points), 0, 1)

print("\nTabla de iteraciones:\n")
print(tabla.to_string(index=False))

# Gráfica de la función y el polinomio interpolante
plt.figure(figsize=(8,6))
plt.plot(x_vals, f(x_vals), label="f(x) = e^{-x} - x", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_ej3.png")
plt.show()

# Gráficas de errores
plt.figure(figsize=(10,6))
plt.plot(tabla["Iteración"], tabla["Error absoluto"], marker='o', label="Error absoluto", color="red")
plt.plot(tabla["Iteración"], tabla["Error relativo"], marker='s', label="Error relativo", color="blue")
plt.plot(tabla["Iteración"], tabla["Error cuadrático"], marker='^', label="Error cuadrático", color="green")
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.title("Convergencia de los errores en el Método de Bisección")
plt.legend()
plt.grid(True)
plt.savefig("errores_ej3.png")
plt.show()
