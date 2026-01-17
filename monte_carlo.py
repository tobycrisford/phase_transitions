import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation

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
            self.lattice[tuple(sample_point)] = -1
        else:
            self.lattice[tuple(sample_point)] = 1


    def run_n_updates(self, n: int, progress: bool = False) -> None:
        if progress:
            iterator = tqdm(range(n))
        else:
            iterator = range(n)
        for _ in iterator:
            self.monte_carlo_update()

    def get_boolean_lattice(self) -> np.ndarray:

        return self.lattice > 0



def animate_model(lattice: Lattice, n_iterations: int, refresh_every: int) -> None:

    fig, ax = plt.subplots()
    img = ax.imshow(lattice.get_boolean_lattice(), cmap='Blues', interpolation='nearest')

    def _update(frame):
        lattice.run_n_updates(refresh_every)
        
        img.set_data(lattice.get_boolean_lattice())
        return [img]

    ani = FuncAnimation(fig, _update, frames=int(n_iterations / refresh_every), interval=1, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':

    test_lattice = Lattice(2, 100, 0.44)

    animate_model(test_lattice, 1000000, 10000)

