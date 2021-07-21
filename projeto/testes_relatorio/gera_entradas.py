import random


def gera_entradas(nome_arquivo, N, M, minVal, maxVal):

    values = ""
    for _ in range(N):
        value_i = random.randint(minVal, maxVal)
        values += str(value_i) + " "

    with open(nome_arquivo, "w") as f:
        f.write(f"{N} {M}\n")
        f.write(values)


if __name__ == "__main__":
    possibleMs = []
    possibleMs += [i for i in range(2, 5)]
    possibleMs += [i for i in range(5, 30, 2)]

    count = 0
    for M in possibleMs:
        gera_entradas(f"in{count}_N.txt", 30, M, 1, 100)
        count += 1

    count = 0
    for N in range(6, 400):
        gera_entradas(f"in{count}_M.txt", N, 5, 1, 100)
        count += 1

    count = 0
    for N in range(5, 19):
        gera_entradas(f"in{count}_M_global.txt", N, 5, 1, 100)
        count += 1

    count = 0
    for M in range(3, 9):
        gera_entradas(f"in{count}_N_global.txt", 14, M, 1, 100)
        count += 1

    count = 0
    M = 5
    for N in range(M, 13):
        gera_entradas(f"in{count}_valor.txt", N, M, 95, 100)
        count += 1