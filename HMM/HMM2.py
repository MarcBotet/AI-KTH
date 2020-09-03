
def create_matrix(num_row, num_column, l):
    mat = []
    for i in range(num_row):
        row = []
        for j in range(num_column):
            row.append(l[num_column * i + j])
        mat.append(row)
    return mat


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def elementwise_multiply(matrix_a, matrix_b):
    return [[a*b for a, b in zip(matrix_a[0], matrix_b)]]


def elementwise_multiply_different_size(alpha, matrix_b):
    mat = []
    for i in matrix_b:
        mat = mat + elementwise_multiply(alpha, i)
    return mat


def mult_observations(matrix, obs):
    mat = []
    for o in range(len(matrix)):
        mat.append([i * obs[o] for i in matrix[o]])
    return mat


def get_max_state(matrix):
    probs = []
    states = []
    for row in matrix:
        maximum = max(row)
        ind = row.index(maximum)
        probs.append(maximum)
        states.append(ind)
    return probs, states



a = [float(x) for x in input().split()]
b = [float(x) for x in input().split()]
pi = [float(x) for x in input().split()]
e = [int(x) for x in input().split()]
emissions = e[1:]

A = create_matrix(int(a[0]), int(a[1]), a[2:])
B = create_matrix(int(b[0]), int(b[1]), b[2:])
PI = create_matrix(int(pi[0]), int(pi[1]), pi[2:])

obs = transpose(B)
At = transpose(A)

alpha = elementwise_multiply(PI, obs[emissions[0]])
probabilities = alpha
states = [None]

for e in emissions[1:]:
    aux = elementwise_multiply_different_size(alpha, At)
    aux = mult_observations(aux, obs[e])
    alpha, s = get_max_state(aux)
    probabilities.append(alpha)
    states.append(s)
    alpha = [alpha]

result = [] # we have to reversed
end_i = len(probabilities) - 1

maximum = max(probabilities[end_i])
ind = probabilities[end_i].index(maximum)
result.append(ind)
state_prev = states[end_i][ind]
result.append(state_prev)

for i in reversed(range(len(probabilities) - 1)):
    if states[i] is None: break
    state_prev = states[i][state_prev]
    result.append(state_prev)

print(' '.join(map(str, list(reversed(result)))))
