def calculate(model, n, emissions):
    A = model.A
    B = model.B
    initial_state = model.pi
    T = len(emissions)
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

    return Model(A,B,initial_state)
