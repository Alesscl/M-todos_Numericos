import numpy as np
import matplotlib.pyplot as plt

# Función original
def f(x):
    return x**3 - 6*x**2 + 11*x - 6

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
        return None, None
    
    iteraciones = []
    errores = []
    c_old = a  # Para calcular errores

    print("\nIteraciones del Método de Bisección:")
    print("Iter |       a       |       b       |       c       |      f(c)      |     Error     ")
    print("-" * 85)

    for i in range(max_iter):
        c = (a + b) / 2
        iteraciones.append(c)
        
        error = abs(c - c_old)
        errores.append(error)

        print(f"{i+1:4d} | {a:.8f} | {b:.8f} | {c:.8f} | {func(c):.8f} | {error:.8e}")

        if abs(func(c)) < tol or error < tol:
            break

        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
        
        c_old = c

    return iteraciones, errores

# Selección de puntos de interpolación
x0 = 1.4
x1 = 1.9
x2 = 2.6
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección con tabla
iteraciones, errores = biseccion(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)

# Gráfica de la función y la interpolación
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 6x^2 + 11x - 6", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")
plt.show()

# Gráfica de convergencia del error absoluto
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(errores)+1), errores, marker='o', linestyle='-', color='red')
plt.yscale("log")  # Escala logarítmica para ver mejor la convergencia
plt.xlabel("Iteración")
plt.ylabel("Error Absoluto")
plt.title("Convergencia del Error Absoluto en el Método de Bisección")
plt.grid(True)
plt.savefig("errores_biseccion.png", dpi=300)
plt.show()
