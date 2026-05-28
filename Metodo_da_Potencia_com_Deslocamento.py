import numpy as np

max_iter = int(input("Digite o número máximo de iterações: "));
n = int(input("Digite a ordem da matriz: "));

q0 = np.array(
    list(map(float, input("Digite o vetor q0: ").split()))
).reshape(n, 1)

print("Vetor q0:")
print(q0)

if np.linalg.norm(q0, 2) != 1:
    print("O vetor inicial q0 não é unitário.")
    q0 = q0 / np.linalg.norm(q0, 2)

A = []
for i in range(n):
    row = list(map(float, input(f"Digite a linha {i} da matriz A: ").split()))
    A.append(row)

A = np.array(A)

print("Matriz A:")
print(A)

q_new = q0
p = (q_new.T @ A @ q_new).item()
for k in range(max_iter + 1):
    print(f"=== Iteração {k}: ===")

    I = np.eye(n)
    w = np.linalg.solve(A - p * I, q_new)
    print("\nVetor w:")
    print(w)

    s = np.linalg.norm(w, 2)
    print("\nValor de s:")
    print(s)

    q_new = w / s
    print(f"\nVetor q_{k+1}:")
    print(q_new)

    autovalor_new = (q_new.T @ A @ q_new).item()
    print(f"\nAutovalor aproximado λ_{k+1}:")
    print(autovalor_new)