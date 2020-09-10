from pprint import pprint
import math
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


def matrix_multiply(matrix_a, matrix_b):
    mat = [[0] * len(matrix_b[0])] * len(matrix_a)
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                mat[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return mat


def elementwise_multiply(matrix_a: list, matrix_b):
    return [[a*b for a, b in zip(matrix_a[0], matrix_b)]]


def calculate(initial_state):
    # alpha-pass
    '''
    obs = transpose(B)
    alpha = elementwise_multiply(initial_state, obs[emissions[0]])
    ct = []

    s = sum(alpha[0])
    c = 1.0/s if s != 0 else 0.0
    alpha = [[c * a for a in alpha[0]]]
    ct.append(c)

    for e in emissions[1:]:
        alpha_aux = matrix_multiply(alpha, A)
        alpha_aux = elementwise_multiply(alpha_aux, obs[e])
        # scale
        s = sum(alpha_aux[0])
        c = 1.0/s if s != 0 else 0.0
        alpha.append([c * a for a in alpha_aux[0]])
        ct.append(c)
    '''

    ct = [0 for _ in range(T)]
    alpha = [[0 for _ in range(n)] for _ in range(T)]
    for i in range(n):
        alpha[0][i] = initial_state[0][i] * B[i][emissions[0]]
        ct[0] = ct[0] + alpha[0][i]

    ct[0] = 1 / ct[0]
    for i in range(n):
        alpha[0][i] = ct[0] * alpha[0][i]

    for t in range(1, T):
        for i in range(n):
            for j in range(n):
                alpha[t][i] = alpha[t][i] + alpha[t-1][j] * A[j][i]
            alpha[t][i] = alpha[t][i] * B[i][emissions[t]]
            ct[t] = ct[t] + alpha[t][i]

        # scale
        ct[t] = 1 / ct[t]
        for i in range(n):
            alpha[t][i] = ct[t] * alpha[t][i]

    # beta-pass

    beta = [[ct[-1] for _ in range(n)] for _ in range(T)]

    for t in reversed(range(T-1)):
        for i in range(n):
            beta[t][i] = 0.0
            for j in range(n):
                beta[t][i] = beta[t][i] + A[i][j] * B[j][emissions[t+1]] * beta[t+1][j]
            beta[t][i] = ct[t] * beta[t][i]

    # compute gamma_t(i,j) and gamma_t(i)

    gamma = [[0 for _ in range(n)] for _ in range(T)]
    gamma_ij = []

    for t in range(T-1):
        gamma_aux = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                gamma_aux[i][j] = alpha[t][i] * A[i][j] * B[j][emissions[t+1]] * beta[t+1][j]
                gamma[t][i] = gamma[t][i] + gamma_aux[i][j]
        gamma_ij.append(gamma_aux)

    for i in range(n):
        gamma[T-1][i] = alpha[T-1][i]

    # re-estimate A, B and pi

    # pi
    initial_state = gamma[0].copy()
    # A
    for i in range(n):
        denom = 0
        for t in range(T-1):
            denom = denom + gamma[t][i]
        for j in range(n):
            numer = 0
            for t in range(T-1):
                numer = numer + gamma_ij[t][i][j]
            A[i][j] = numer / denom if denom != 0 else 0

    # B
    for i in range(n):
        denom = 0
        for t in range(T):
            denom = denom + gamma[t][i]
        for j in range(m):
            numer = 0
            for t in range(T):
                if emissions[t] == j:
                    numer = numer + gamma[t][i]
            B[i][j] = numer / denom if denom != 0 else 0

    # compute log(P(O|lambda))

    logprob = 0
    for i in range(T):
        logprob = logprob + math.log(ct[i])
    logprob = -logprob

    return logprob, [initial_state]


maxIters = 30

a = [float(x) for x in input().split()]
b = [float(x) for x in input().split()]
pi = [float(x) for x in input().split()]
e = [int(x) for x in input().split()]
T = e[0]
emissions = e[1:]

A = create_matrix(int(a[0]), int(a[1]), a[2:])
n = int(a[0])
B = create_matrix(int(b[0]), int(b[1]), b[2:])
m = int(b[1])
PI = create_matrix(int(pi[0]), int(pi[1]), pi[2:])

# to iterate or not iterate
logprob = 1
oldlogprob = -math.inf
iters = 0
while iters < maxIters and logprob > oldlogprob:
    iters = iters + 1
    if iters != 1: oldlogprob = logprob
    logprob, PI = calculate(PI)

A = ' '.join(map(lambda x: ' '.join(map(str, x)), A))
print(n, n, A)
B = ' '.join(map(lambda x: ' '.join(map(str, x)), B))
print(n, m, B)




