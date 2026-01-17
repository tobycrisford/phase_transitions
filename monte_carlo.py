import numpy as np
from tqdm import tqdm

class Lattice:

    STATES = np.array([-1, 1])

    def __init__(self, dimension: int, N: int, beta: float):

        self.lattice = (np.random.rand(*[N for _ in range(dimension)]) > 0.5) * 2 - 1
        self.beta = beta
        self.N = N
        self.dimension = dimension

    def _compute_energy_term(self, sample_point: np.ndarray, dim_index: int, neighbour_diff: int) -> float:

        sample_point[dim_index] = (sample_point[dim_index] + neighbour_diff) % self.N
        energy_vals = self.STATES * self.lattice[tuple(sample_point)]
        sample_point[dim_index] = (sample_point[dim_index] - neighbour_diff) % self.N
        return -1 * energy_vals

    def monte_carlo_update(self) -> None:

        sample_point = np.random.randint(0, self.N, size=self.dimension)

        energy_vals = np.zeros(2)
        for d in range(self.dimension):
            energy_vals += self._compute_energy_term(sample_point, d, 1)
            energy_vals += self._compute_energy_term(sample_point, d, -1)

        probs = np.exp(-1 * self.beta * energy_vals)
        probs /= np.sum(probs)

        if np.random.rand() < probs[0]:
            self.lattice[sample_point] = -1
        else:
            self.lattice[sample_point] = 1


    def run_n_updates(self, n: int, progress: bool = False) -> None:
        for _ in tqdm(range(n)):
            self.monte_carlo_update()


if __name__ == '__main__':

    test_lattice = Lattice(2, 1000, 0.1)

    test_lattice.run_n_updates(1000000, progress=True)

    breakpoint()

