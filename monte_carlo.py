import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation

class Lattice:

    ISING_STATES = np.array([0.0, np.pi])
    O2_STATES = np.linspace(0, 2*np.pi, endpoint=False, num=100)

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

        rgb_data = np.zeros(tuple(self.N for _ in range(self.dimension)) + (3, ))
        rgb_data[..., 0] = (np.cos(self.lattice) + 1) / 2
        rgb_data[..., 1] = (np.sin(self.lattice) + 1) / 2
        rgb_data[..., 2] = 0.5
        
        if self.dimension == 2:
            return rgb_data
        elif self.dimension == 1:
            img_lattice = np.zeros((self.N, self.N, 3))
            img_lattice[:, :, :] = rgb_data
            return img_lattice
        else:
            raise NotImplementedError()

    
    def _find_closest_diff(self, theta_a: float, theta_b: float) -> float:

        diff = (theta_a - theta_b) % (2 * np.pi)
        if diff > np.pi:
            return diff - 2 * np.pi
        else:
            return diff
    
    def check_for_vortex(self, x: int, y: int) -> int:

        if not self.o_dimension == 2:
            raise Exception('Vortices only make sense for the O2 model')

        diff_a = self._find_closest_diff(self.lattice[(x+1) % self.N, y], self.lattice[x, y])
        diff_b = self._find_closest_diff(self.lattice[(x+1) % self.N, (y+1) % self.N], self.lattice[(x+1) % self.N, y])
        diff_c = self._find_closest_diff(self.lattice[x, (y+1)%self.N], self.lattice[(x+1) % self.N, (y+1) % self.N])
        diff_d = self._find_closest_diff(self.lattice[x, y], self.lattice[x, (y+1) % self.N])

        winding_number = round((diff_a + diff_b + diff_c + diff_d) / (2 * np.pi))

        return winding_number

    
    def find_vortices(self) -> np.ndarray:

        if not self.o_dimension == 2:
            raise Exception('Vortices only make sense for the O2 model')

        vortices = []
        for x in range(self.N):
            vortices.append([])
            for y in range(self.N):
                vortices[x].append(self.check_for_vortex(x, y))

        return np.array(vortices)





def animate_model(lattice: Lattice, n_iterations: int, refresh_every: int) -> None:

    if lattice.o_dimension == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
    else:
        fig, ax1 = plt.subplots()
    
    img = ax1.imshow(lattice.get_image_lattice())
    if lattice.o_dimension == 2:
        vortex_img = ax2.imshow(lattice.find_vortices(), cmap='Blues', interpolation='nearest')
    else:
        vortex_img = None

    def _update(frame):
        lattice.run_n_updates(refresh_every)
        
        updated = []
        img.set_data(lattice.get_image_lattice())
        updated.append(img)
        if vortex_img:
            vortex_img.set_data(lattice.find_vortices())
            updated.append(vortex_img)
        
        return updated

    ani = FuncAnimation(fig, _update, interval=1, blit=True, repeat=False, cache_frame_data=False)
    plt.show()


if __name__ == '__main__':

    test_lattice = Lattice(2, 100, 10.0, 2)

    animate_model(test_lattice, 1000000, 10000)

