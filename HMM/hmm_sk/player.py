import math
import sys
from pprint import pprint

import numpy as np
from constants import *
from player_controller_hmm import PlayerControllerHMMAbstract

STEP_OBSERVATION = 110
epsilon = sys.float_info.epsilon
N_HIDDEN = 1


def calculate(model, n, m, emissions, max_iters=1):
    A = model.A
    B = model.B
    initial_state = model.pi
    T = len(emissions)

    logprob = 1
    oldlogprob = -math.inf
    iters = 0
    while iters < max_iters and logprob > oldlogprob:
        ct = [0 for _ in range(T)]
        alpha = [[0 for _ in range(n)] for _ in range(T)]

        for i in range(n):
            alpha[0][i] = initial_state[0][i] * B[i][emissions[0]]
            ct[0] = ct[0] + alpha[0][i]

        ct[0] = 1 / (ct[0] + epsilon)
        for i in range(n):
            alpha[0][i] = ct[0] * alpha[0][i]

        for t in range(1, T):
            for i in range(n):
                for j in range(n):
                    alpha[t][i] = alpha[t][i] + alpha[t - 1][j] * A[j][i]
                alpha[t][i] = alpha[t][i] * B[i][emissions[t]]
                ct[t] = ct[t] + alpha[t][i]

            # scale
            ct[t] = 1 / (ct[t] + epsilon)
            for i in range(n):
                alpha[t][i] = ct[t] * alpha[t][i]

        # beta-pass

        beta = [[ct[-1] for _ in range(n)] for _ in range(T)]

        for t in reversed(range(T - 1)):
            for i in range(n):
                beta[t][i] = 0.0
                for j in range(n):
                    beta[t][i] = beta[t][i] + A[i][j] * B[j][emissions[t + 1]] * beta[t + 1][j]
                beta[t][i] = ct[t] * beta[t][i]

        # compute gamma_t(i,j) and gamma_t(i)

        gamma = [[0 for _ in range(n)] for _ in range(T)]
        gamma_ij = []

        for t in range(T - 1):
            gamma_aux = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    gamma_aux[i][j] = alpha[t][i] * A[i][j] * B[j][emissions[t + 1]] * beta[t + 1][j]
                    gamma[t][i] = gamma[t][i] + gamma_aux[i][j]
            gamma_ij.append(gamma_aux)

        for i in range(n):
            gamma[T - 1][i] = alpha[T - 1][i]

        # re-estimate A, B and pi

        # pi
        initial_state = [gamma[0].copy()]
        # A
        for i in range(n):
            denom = 0
            for t in range(T - 1):
                denom = denom + gamma[t][i]
            for j in range(n):
                numer = 0
                for t in range(T - 1):
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
        # print("logprob ", logprob, oldlogprob)
        iters = iters + 1
        if iters != 1: oldlogprob = logprob

    print("iteracion: ", iters)
    return A, B, initial_state


def elementwise_multiply(matrix_a, matrix_b):
    return [[a * b for a, b in zip(matrix_a[0], matrix_b)]]


def transpose(matrix):
    return [list(i) for i in zip(*matrix)]
    # return list(map(list, zip(*matrix)))


def matrix_multiply(matrix_a, matrix_b):
    '''mat = [[0] * len(matrix_b[0])] * len(matrix_a)
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                mat[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return mat'''
    # comprehension list are faster
    return [[sum(a * b for a, b in zip(a_row, b_col)) for b_col in zip(*matrix_b)] for a_row in matrix_a]


def alpha_pass(fish, model):
    obs = transpose(model.B)
    alpha = elementwise_multiply(model.pi, obs[fish[0]])

    for e in fish[1:]:
        alpha = matrix_multiply(alpha, model.A)
        alpha = elementwise_multiply(alpha, obs[e])

    return sum(alpha[0])


def row_stochastic_matrix(n, precision=1000):
    matrix = [(1 / n) + np.random.rand() / precision for _ in range(n)]
    s = sum(matrix)
    return [m / s for m in matrix]


class Model:
    def __init__(self, species, emissions):
        self.pi = [rowStochasticMatrix(species)]
        self.A = [rowStochasticMatrix(species) for _ in range(species)]
        self.B = [rowStochasticMatrix(emissions) for _ in range(species)]
        '''
        self.pi = np.random.dirichlet(np.ones(species),size=1)
        self.A = [np.random.dirichlet(np.ones(species),size=1)[0] for _ in range(species)]
        self.B = [np.random.dirichlet(np.ones(emissions),size=1)[0] for _ in range(species)]
        self.pi = [[1 / species for _ in range(species)]]
        self.A = [[1 / species for _ in range(species)] for _ in range(species)]
        self.B = [[1 / emissions for _ in range(emissions)] for _ in range(species)]
        '''

    def set_A(self, A):
        self.A = A

    def set_B(self, B):
        self.B = B

    def set_pi(self, pi):
        self.pi = pi


class PlayerControllerHMM(PlayerControllerHMMAbstract):

    def update_model(self, model_id):
        pprint(self.models_fish[model_id].B)
        A, B, pi = calculate(self.models_fish[model_id], N_HIDDEN, N_EMISSIONS, self.obs, max_iters=30)
        self.models_fish[model_id].set_A(A)
        self.models_fish[model_id].set_B(B)
        self.models_fish[model_id].set_pi(pi)
        print("*************************")
        pprint(B)
        # print("model_id: ", model_id)

    # else:
    # A, B, pi = calculate(self.models_fish[model_id], N_SPECIES, N_EMISSIONS, obs)

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.seen_fishes = set()
        self.seen_species = set()

        self.models_fish = [Model(N_HIDDEN, N_EMISSIONS) for _ in range(N_SPECIES)]

        self.fishes = [(i, []) for i in range(N_FISH)]

        self.count_guess_total = 0

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        for i in range(len(self.fishes)):
            self.fishes[i][1].append(observations[i])

        if step < STEP_OBSERVATION:
            return None
        else:
            index_fish, obs = self.fishes.pop()
            index_type = 0
            max = 0
            for model, j in zip(self.models_fish, range(N_SPECIES)):
                m = alpha_pass(obs, model)
                if m > max:
                    max = m
                    index_type = j
            self.obs = obs
            # print("prob: ", math.log(max))
            return index_fish, index_type

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        # print(correct, true_type)
        self.count_guess_total += 1
        # print(self.count_guess_total)
        if true_type not in self.seen_species:
            self.update_model(true_type)
            self.seen_species.add(true_type)
