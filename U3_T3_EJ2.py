import numpy as np
import matplotlib.pyplot as plt

# =========================
#  Datos del ejercicio 2
# =========================
A = np.array([[ 8,  2, -1,  0,  0,  0],
              [ 3, 15, -2,  1,  0,  0],
              [ 0, -2, 12,  2, -1,  0],
              [ 0,  1, -1,  9, -2,  1],
              [ 0,  0, -2,  3, 14,  1],
              [ 0,  0,  0,  1, -2, 10]], dtype=float)

b = np.array([10, 24, -18, 16, -9, 22], dtype=float)

tolerancia = 1e-6
max_iter = 100

# Solución exacta para comparar errores
sol_exacta = np.linalg.solve(A, b)

# =========================
#  Método de Jacobi
# =========================
def jacobi(A, b, tol, max_iter):
    n = len(A)
    x = np.zeros(n)  # Aproximación inicial
    errores_abs = []
    errores_rel = []
    errores_cuad = []

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i, i]

        # Calcular errores contra la solución exacta
        error_vec = x_new - sol_exacta
        error_abs  = np.linalg.norm(error_vec, ord=1)
        error_rel  = error_abs / np.linalg.norm(sol_exacta, ord=1)
        error_cuad = np.linalg.norm(error_vec, ord=2)

        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)

        # Imprimir errores de la iteración (tabla en consola)
        print(f"Iteración {k+1:>3}:  Abs = {error_abs:.6e} | Rel = {error_rel:.6e} | L2 = {error_cuad:.6e}")

        # Criterio de convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            x = x_new
            break

        x = x_new

    return x, errores_abs, errores_rel, errores_cuad, k+1

# =========================
#  Ejecutar Jacobi
# =========================
sol_aprox, errores_abs, errores_rel, errores_cuad, iteraciones = jacobi(A, b, tolerancia, max_iter)

# =========================
#  Graficar los errores
# =========================
plt.figure(figsize=(8,6))
plt.plot(range(1, iteraciones+1), errores_abs,  label="Error absoluto",  marker='o')
plt.plot(range(1, iteraciones+1), errores_rel,  label="Error relativo",  marker='s')
plt.plot(range(1, iteraciones+1), errores_cuad, label="Error cuadrático", marker='d')
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Convergencia de los errores en el método de Jacobi (Ejercicio 2)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig("EJ2_errores_jacobi.png", dpi=150)
plt.show()

# =========================
#  Mostrar soluciones
# =========================
print("\n================  Resultados  ================")
print(f"Iteraciones: {iteraciones}")
print(f"x (Jacobi)  : {sol_aprox}")
print(f"x (solve)   : {sol_exacta}")
print(f"||dif||2    : {np.linalg.norm(sol_aprox - sol_exacta, ord=2):.12e}")
