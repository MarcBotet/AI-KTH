
def create_matrix(num_row, num_column, l):
    mat = []
    for i in range(num_row):
        row = []
        for j in range(num_column):
            row.append(l[num_column * i + j])
        mat.append(row)
    return mat


def multiply_matrix(matrix_a, matrix_b):
    mat = [[0] * len(matrix_b[0])] * len(matrix_a)
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                mat[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return mat


a = [float(x) for x in input().split()]
b = [float(x) for x in input().split()]
pi = [float(x) for x in input().split()]

A = create_matrix(int(a[0]), int(a[1]), a[2:])
B = create_matrix(int(b[0]), int(b[1]), b[2:])
PI = create_matrix(int(pi[0]), int(pi[1]), pi[2:])

result = multiply_matrix(multiply_matrix(PI, A), B)

result_print = ' '.join(map(lambda x: ' '.join(map(str, x)), result))
print(len(result), len(result[0]), result_print)

