def create_matrix(num_row, num_column, l):
    mat = []
    for i in range(num_row):
        row = []
        for j in range(num_column):
            row.append(l[num_column * i + j])
        mat.append(row)
    return mat


def transpose(matrix):
    return [list(i) for i in zip(*matrix)]
    # return list(map(list, zip(*matrix)))


def matrix_multiply2(matrix_a, matrix_b):
    mat = [[0] * len(matrix_b[0])] * len(matrix_a)
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                mat[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return mat


def matrix_multiply(X, Y):
    return [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)] for X_row in X]


def elementwise_multiply2(matrix_a, matrix_b):
    return [[a[0] * b[0]] for a, b in zip(matrix_a, matrix_b)]


def elementwise_multiply(matrix_a, matrix_b):
    return [[a * b for a, b in zip(matrix_a[0], matrix_b)]]


a = [float(x) for x in input().split()]
b = [float(x) for x in input().split()]
pi = [float(x) for x in input().split()]
e = [int(x) for x in input().split()]
emissions = e[1:]

A = create_matrix(int(a[0]), int(a[1]), a[2:])
B = create_matrix(int(b[0]), int(b[1]), b[2:])
PI = create_matrix(int(pi[0]), int(pi[1]), pi[2:])

obs = transpose(B)
alpha = elementwise_multiply(PI, obs[emissions[0]])

for e in emissions[1:]:
    alpha = matrix_multiply(alpha, A)
    alpha = elementwise_multiply(alpha, obs[e])

print(sum(alpha[0]))
