# implementation of MARKOV CHAIN MONTE-CARLO optimization on discrete grid


# libraries
import math
import argparse
from scipy.stats import bernoulli
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt


# helpers by staff
def get_cmdline_parser():
    """
    method to help parse info from cmd line
    :return: the parser itself
    """
    parser = argparse.ArgumentParser(
        description='Run MCMC on a discrete n*n configuration space.')
    parser.add_argument('n', type=int, default=5, nargs='?',
                        help='number of rows/columns of grid')
    parser.add_argument('m', type=int, default=1000, nargs='?',
                        help='number of iteration of MCMC optimization')
    parser.add_argument('kT', type=float, default=1.0, nargs='?',
                        help='kT - the denominator for the metropolis '
                             'criterion, (Boltzmann constant times '
                             'temperature)')
    return parser


def is_valid(c: tuple, n: int):
    """ Return True if c is a valid 2-D coordinate on an n x n grid
   with 0-based indices """
    if len(c) != 2:
        return False
    return (c[0] >= 0 and c[1] >= 0 and c[0] < n and c[1] < n)


def get_p_accept_metropolis(dE, kT, p_forward, p_backward):
    """
    return the probability to accept the metropolis criteria
    for the specified conditions
    dE - change in energy from current to proposed configuration
    kT - the factor of Boltzmann constant (kB) and the temperature (T)
    p_forward - probability to propose a move from current to proposed
    configuration p_backward - probability to propose a move from proposed to
    current configuration """

    p = (p_backward / p_forward) * (math.exp((-1 * dE) / kT))
    return min(p, 1.0)


def E(c: tuple):
    """
    method to get Energy levels of coordinates C
    :param c: tuple of (x,y) on grid
    :return: energy levels
    """
    assert (len(c) == 2)
    return 1.0 * c[0] + 0.5 * c[1]


def get_neighbours(c: tuple, n: int):
    """
    get up/down/left/right neighbours on an n x n grid with 0-based indices
    """
    assert (is_valid(c, n))
    ret_value = []
    if c[0] > 0:
        ret_value.append((c[0] - 1, c[1]))
    if c[0] < n - 1:
        ret_value.append((c[0] + 1, c[1]))
    if c[1] > 0:
        ret_value.append((c[0], c[1] - 1))
    if c[1] < n - 1:
        ret_value.append((c[0], c[1] + 1))
    return ret_value


def generate_random_configuration_space(n: int):
    """
    creation of configuration space
    :param n: number of rows/cols
    :return: configuration matrix
    """
    configuration = np.empty_like(np.zeros(shape=(n, n), dtype=object))
    for i in range(n):
        for j in range(n):
            configuration[i][j] = (i, j)
    return configuration


def randomize_start(visitor_grid, n: int):
    """
    method to randomisze a start position
    :param visitor_grid: grid of visiting, to mark that we visied the first
    picked one
    :param n: row/col amount
    :return:
    """
    random_start_position = np.random.randint(low=0, high=n, size=(1, 2))[0]
    start_row, start_col = random_start_position[0], random_start_position[1]
    visitor_grid[start_row][start_col] += 1
    return start_row, start_col


def get_probability_from_neigs_length(cords: tuple, n: int):
    """
    get probability from picking a neighbour of the coordinate c
    uniform probability over the amount of neighbours
    :param cords: coordinates
    :param n: # of rows/cols of matrix
    :return: uniform probability to pic a neighbouring coordinate
    """
    return 1 / len(get_neighbours(cords, n))


def calculate_change_in_energy(current: tuple, propositional: tuple,
                               energy_grid):
    """
    calculate change of energy from current configuration to propositional
    :param energy_grid: energy grid
    :param current: current coordinate
    :param propositional: propositional coordinate
    :return: difference of energy form cor to prop
    """
    return energy_grid[propositional] - energy_grid[current]


def accept_or_reject(curr: tuple, prop: tuple, metropolis_p: float,
                     visitor_grid):
    """
    randomizing a pick by metropolis probability weather to make a step to p
    or not.
    :param curr: current cord to move from
    :param prop: current cord to move to
    :param metropolis_p: probability of metropolis calculated before
    :param visitor_grid: visitor grid to mark
    :return: none
    """
    # decision = np.random.binomial(n=1, p=metropolis_p, size=1)[0]
    decision = bernoulli.rvs(metropolis_p)
    if decision:
        # accept
        curr = prop
    visitor_grid[curr[0]][curr[1]] += 1


def create_energy_grid(n: int):
    """
    method to create an energy grid for the n*n grid
    :param n: amount of rows, cols
    """
    grid = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            grid[i, j] = E((i, j))
    return grid


def run_MCMC(n: int, m: int, kT: float):
    """
    running MCMC according to given values
    :param n: number of rows/cols
    :param m: number of rotations
    :param kT: given boltzman value
    :return: vistior grid
    """
    visitor_grid = np.zeros(shape=(n, n), dtype=int)
    energy_grid = create_energy_grid(n)
    current_row, current_col = randomize_start(visitor_grid, n)
    # visited first pick
    for iteration in range(m - 1):
        neigs = get_neighbours((current_row, current_col), n)
        random_proposition = neigs[np.random.randint(0, high=len(neigs))]
        proposition_row, proposition_col = random_proposition[0], \
                                           random_proposition[1]
        c, p = (current_row, current_col), (proposition_row, proposition_col)
        p_forward = get_probability_from_neigs_length(c, n)
        p_backward = get_probability_from_neigs_length(p, n)
        dE = calculate_change_in_energy(c, p, energy_grid)
        metropolis_p = get_p_accept_metropolis(dE, kT, p_forward, p_backward)
        if bernoulli.rvs(metropolis_p):
            current_row = proposition_row
            current_col = proposition_col
            visitor_grid[current_row, current_col] += 1
    return visitor_grid


def kT_heat_map(vg, kT, text):
    plt.figure()

    log_vg = kT * (np.max(np.log1p(vg)) - np.log1p(vg))
    sbn.heatmap(log_vg, annot=True, fmt=".2f",
                linewidths=.5)
    plt.xlabel("index of row")
    plt.ylabel("index of column")
    if text:
        plt.title("frequencies of coordinate visiting of m=500,000, "
                  "with kT " + str(kT) + ", " + text)
    else:
        plt.title("frequencies of coordinate visiting of m=500,000, "
                  "with log scale")
    plt.show()


def log_heat_map(vg, text):
    plt.figure()

    log_vg = np.max(np.log1p(vg)) - np.log1p(vg)
    sbn.heatmap(log_vg, annot=True, fmt=".2f",
                linewidths=.5)
    plt.xlabel("index of row")
    plt.ylabel("index of column")
    if text:
        plt.title("frequencies of coordinate visiting of m=500,000, "
                  "with log. " + text)
    else:
        plt.title("frequencies of coordinate visiting of m=500,000, "
                  "with log scale")
    plt.show()


def design_heat_map(vg, m):
    """
    design heat map by iteration and visitor grid
    :param vg: visitor grid
    :param m: number of iteration
    :return: none
    """
    hp_divided = vg / m

    plt.figure()

    sbn.heatmap(hp_divided, annot=True, fmt=".2f", linewidths=.5)
    plt.xlabel("index of row")
    plt.ylabel("index of column")
    plt.title("frequencies of coordinate visiting of m:" + str(m))
    plt.show()


def part_1_algorithm():
    """
    part 1 algorithm of ex3
    :return:none
    """
    parser = get_cmdline_parser()
    args = parser.parse_known_args()[0]
    n, m, kT = args.n, args.m, args.kT
    configurations = generate_random_configuration_space(n)
    # q2-5
    for m in [10, 100, 500000]:
        visitor_grid = run_MCMC(n, m, kT)
        design_heat_map(visitor_grid, m)
        if m == 500000:
            log_heat_map(visitor_grid, None)
    # q7-10
    m = 500000
    for kT in [0.1, 1.0, 2.0, 50.0]:
        visitor_grid = run_MCMC(n, m, kT)
        log_heat_map(visitor_grid, "kT: " + str(kT) + ".")
        kT_heat_map(visitor_grid, kT, "\n(multiplied by)")


if __name__ == '__main__':
    part_1_algorithm()
