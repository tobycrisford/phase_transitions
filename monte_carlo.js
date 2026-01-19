function random_ising_lattice_site() {
    return (Math.random() > 0.5) * Math.PI;
}

function random_xy_lattice_site() {
    return Math.random() * 2.0 * Math.PI;
}

function random_lattice(dimension, N, site_generator) {
    const lattice = [];
    if (dimension === 1) {
        for (let i = 0;i < N;i++) {
            lattice.push(site_generator());
        }
    }
    else if (dimension > 1) {
        for (let i = 0;i < N;i++) {
            lattice.push(random_lattice(dimension - 1, N, site_generator));
        }
    }
    else {
        throw new Error('Bad dimension arg');
    }
}

const O_DIM_SPREAD = 100;

function get_array_value(arr, indices) {
    let current = arr;
    for (const index of indices) {
        current = current[index];
    }
    return current;
}

function set_array_value(arr, indices, val) {
    let current = arr;
    for (let i = 0;i < arr.length - 1;i++) {
        current = current[index];
    }
    current[indices[indices.length - 1]] = val;
}

// This function taken verbatim from stackexchange
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function sample_from_dist(dist) {
    const r = Math.random();
    let cumsum = 0.0
    for (let i = 0;i < dist.length;i++) {
        cumsum += dist[i];
        if (cumsum > r) {
            return i;
        }
    }
    if (1 - cumsum > 10**(-9)) {
        throw new Error('Something has gone wrong with dist sample');
    }
    return dist.length - 1;
}

class Lattice {
    constructor(dimension, N, beta, o_dimension) {

        if (o_dimension === 1) {
            this.lattice = random_lattice(dimension, N, random_ising_lattice_site);
            this.states = [0.0, Math.PI];
        }
        else if (o_dimension === 2) {
            this.lattice = random_lattice(dimension, N, random_xy_lattice_site);
            this.states = [];
            for (let i = 0;i < O_DIM_SPREAD;i++) {
                this.states.push(i * ((2.0 * Math.PI) / O_DIM_SPREAD));
            }
        }
        else {
            throw new Error('o_dimension currently only supports 1 or 2');
        }

        this.beta = beta;
        this.N = N;
        this.dimension = dimension;
        this.o_dimension = o_dimension;
    }

    _increment_energy_term(sample_point, dim_index, neighbour_diff, energy_values) {
        sample_point[dim_index] = (sample_point[dim_index] + neighbour_diff) % this.N;
        for (let i = 0;i < this.energy_values.length;i++) {
            energy_values[i] -= Math.cos(this.states[i] - get_array_value(this.lattice, sample_point));
        }
        sample_point[dim_index] = (sample_point[dim_index] - neighbour_diff) % this.N;
    }

    monte_carlo_update() {
        const sample_point = Array(this.dimension).fill(0);
        for (let i = 0;i < sample_point.length;i++) {
            sample_point[i] = getRandomInt(0, this.N);
        }

        const energy_values = Array(this.states.length).fill(0.0);
        for (let d = 0;d < this.dimension;d++) {
            this._increment_energy_term(sample_point, d, 1, energy_values);
            this._increment_energy_term(sample_point, d, -1, energy_values);
        }

        let prob_sum = 0.0;
        for (let i = 0;i < energy_values.length;i++) {
            energy_values[i] = Math.exp(-1 * this.beta * energy_values[i]);
            prob_sum += energy_values[i];
        }
        for (let i = 0;i < energy_values.length;i++) {
            energy_values[i] /= prob_sum;
        }

        set_array_value(this.lattice, sample_point, this.states[sample_from_dist(energy_values)]);
    }

    run_n_updates(n) {
        for (let i = 0;i < n;i++) {
            this.monte_carlo_update();
        }
    }
}