import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation

class Lattice:

    ISING_STATES = np.array([0.0, np.pi])
    O2_STATES = np.linspace(0, 2*np.pi, endpoint=False, num=20)

    def __init__(self, dimension: int, N: int, beta: float, o_dimension: int):

        if o_dimension == 1:
            self.lattice = (np.random.rand(*[N for _ in range(dimension)]) > 0.5) * np.pi
            self.states = self.ISING_STATES
        elif o_dimension == 2:
            self.lattice = np.random.rand(*[N for _ in range(dimension)]) * 2 * np.pi
            self.states = self.O2_STATES
        else:
            raise NotImplementedError()

        self.beta = beta
        self.N = N
        self.dimension = dimension
        self.o_dimension = o_dimension

    def _compute_energy_term(self, sample_point: np.ndarray, dim_index: int, neighbour_diff: int) -> float:

        sample_point[dim_index] = (sample_point[dim_index] + neighbour_diff) % self.N
        energy_vals = -1 * np.cos(self.states - self.lattice[tuple(sample_point)])
        sample_point[dim_index] = (sample_point[dim_index] - neighbour_diff) % self.N
        return energy_vals

    def monte_carlo_update(self) -> None:

        sample_point = np.random.randint(0, self.N, size=self.dimension)

        energy_vals = np.zeros(len(self.states))
        for d in range(self.dimension):
            energy_vals += self._compute_energy_term(sample_point, d, 1)
            energy_vals += self._compute_energy_term(sample_point, d, -1)

        probs = np.exp(-1 * self.beta * energy_vals)
        probs /= np.sum(probs)

        self.lattice[tuple(sample_point)] = self.states[np.random.choice(len(self.states), p=probs)]


    def run_n_updates(self, n: int, progress: bool = False) -> None:
        if progress:
            iterator = tqdm(range(n))
        else:
            iterator = range(n)
        for _ in iterator:
            self.monte_carlo_update()

    def get_image_lattice(self) -> np.ndarray:

        if self.dimension == 2:
            return self.lattice
        elif self.dimension == 1:
            img_lattice = np.zeros((self.N, self.N))
            img_lattice[:, :] = self.lattice
            return img_lattice
        else:
            raise NotImplementedError()



def animate_model(lattice: Lattice, n_iterations: int, refresh_every: int) -> None:

    fig, ax = plt.subplots()
    img = ax.imshow(lattice.get_image_lattice(), cmap='Blues', interpolation='nearest')

    def _update(frame):
        lattice.run_n_updates(refresh_every)
        
        img.set_data(lattice.get_image_lattice())
        return [img]

    ani = FuncAnimation(fig, _update, frames=int(n_iterations / refresh_every), interval=1, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':

    test_lattice = Lattice(2, 100, 10, 2)

    animate_model(test_lattice, 1000000, 10000)

