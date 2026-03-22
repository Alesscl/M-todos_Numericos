import numpy as np

# =========================
#  Datos del ejercicio 3 (A no es cuadrada: 6x7)
# =========================
A = np.array([[12, -2,  1,  0,  0,  0,  0],
              [-3, 18, -4,  2,  0,  0,  0],
              [ 1, -2, 16, -1,  1,  0,  0],
              [ 0,  2, -1, 11, -3,  1,  0],
              [ 0,  0, -2,  4, 15, -2,  1],
              [ 0,  0,  0,  1, -3,  2, 13]], dtype=float)

b = np.array([20, 35, -5, 19, -12, 25], dtype=float)

# Esto DEBE lanzar error: A no es cuadrada y solve exige matriz cuadrada.
sol_exacta = np.linalg.solve(A, b)

# Si llegas a ver este print, algo anduvo raro:
print("No deberías ver esto; solve(A,b) debe fallar con matriz no cuadrada.")