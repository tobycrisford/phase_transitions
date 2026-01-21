const mod = (n, m) => ((n % m) + m) % m

function random_ising_lattice_site() {
    return (Math.random() > 0.5) * Math.PI;
}

function random_xy_lattice_site() {
    return Math.random() * 2.0 * Math.PI;
}

function generate_lattice(dimension, N, site_generator) {
    const lattice = [];
    if (dimension === 1) {
        for (let i = 0;i < N;i++) {
            lattice.push(site_generator());
        }
    }
    else if (dimension > 1) {
        for (let i = 0;i < N;i++) {
            lattice.push(generate_lattice(dimension - 1, N, site_generator));
        }
    }
    else {
        throw new Error('Bad dimension arg');
    }

    return lattice;
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
    for (let i = 0;i < indices.length - 1;i++) {
        current = current[indices[i]];
    }
    current[indices[indices.length - 1]] = val;
}

// This function taken verbatim from stackexchange (and fixed!)
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min)) + min;
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

        this.set_lattice_params(dimension, N, o_dimension);
        this.beta = beta;
    }

    set_lattice_params(dimension, N, o_dimension) {
        this.vortices = null;
        if (o_dimension === 1) {
            this.lattice = generate_lattice(dimension, N, random_ising_lattice_site);
            this.states = [0.0, Math.PI];
        }
        else if (o_dimension === 2) {
            this.lattice = generate_lattice(dimension, N, random_xy_lattice_site);
            this.states = [];
            for (let i = 0;i < O_DIM_SPREAD;i++) {
                this.states.push(i * ((2.0 * Math.PI) / O_DIM_SPREAD));
            }
            if (dimension === 2) {
                this.vortices = generate_lattice(dimension, N, () => 0);
            }
        }
        else {
            throw new Error('o_dimension currently only supports 1 or 2');
        }
        this.N = N;
        this.dimension = dimension;
        this.o_dimension = o_dimension;
    }

    _increment_energy_term(sample_point, dim_index, neighbour_diff, energy_values) {
        sample_point[dim_index] = mod((sample_point[dim_index] + neighbour_diff), this.N);
        for (let i = 0;i < energy_values.length;i++) {
            energy_values[i] -= Math.cos(this.states[i] - get_array_value(this.lattice, sample_point));
        }
        sample_point[dim_index] = mod((sample_point[dim_index] - neighbour_diff), this.N);
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

    _find_closest_diff(theta_a, theta_b) {
        const diff = mod((theta_a - theta_b), (2 * Math.PI));
        if (diff > Math.PI) {
            return diff - 2 * Math.PI;
        }
        else {
            return diff;
        }
    }

    check_for_vortex(x, y) {
        if (!( (this.dimension === 2) && (this.o_dimension == 2))) {
            throw new Error('Vortices only make sense for 2D XY model');
        }

        const diff_a = this._find_closest_diff(this.lattice[mod((x+1), this.N)][y], this.lattice[x][y]);
        const diff_b = this._find_closest_diff(this.lattice[mod((x+1), this.N)][mod((y+1), this.N)], this.lattice[mod((x+1), this.N)][y]);
        const diff_c = this._find_closest_diff(this.lattice[x][mod((y+1), this.N)], this.lattice[mod((x+1), this.N)][mod((y+1), this.N)]);
        const diff_d = this._find_closest_diff(this.lattice[x][y], this.lattice[x][mod((y+1), this.N)]);

        const winding_number = Math.round((diff_a + diff_b + diff_c + diff_d) / (2 * Math.PI));

        return winding_number
    }

    update_vortices() {
        if (!(this.dimension === 2 && this.o_dimension == 2)) {
            throw new Error('Vortices only make sense for 2D XY model');
        }

        this.max_vortex = 0;
        for (let x = 0;x < this.N;x++) {
            for (let y = 0;y < this.N;y++) {
                this.vortices[x][y] = this.check_for_vortex(x, y);
                if (Math.abs(this.vortices[x][y]) > this.max_vortex) {
                    this.max_vortex = Math.abs(this.vortices[x][y]);
                }
            }
        }
    }
}


function animate_model(lattice, canvas, vortex_canvas, refresh_every) {
    
    if (!((lattice.dimension === 1) || (lattice.dimension === 2))) {
        throw new Error('Cant draw lattice with dimensions above 1 or 2!')
    }
    
    const ctx = canvas.getContext('2d');
    const vortex_ctx = vortex_canvas.getContext('2d');
    const width = lattice.N;
    const height = lattice.N;
    
    const imageData = ctx.createImageData(width, height);
    const vortexImageData = vortex_ctx.createImageData(width, height);
    const data = imageData.data;
    const vortex_data = vortexImageData.data;
    
    function updateAndDraw() {
        lattice.run_n_updates(refresh_every);
        if (lattice.vortices !== null) {
            lattice.update_vortices();
        }

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                let lattice_site = 0.0;
                if (lattice.dimension === 1) {
                    lattice_site = lattice.lattice[x];
                }
                else {
                    lattice_site = lattice.lattice[x][y];
                }
    
                data[i] = 255 * ((Math.cos(lattice_site) + 1) / 2); // Red
                data[i + 1] = 255 * ((Math.sin(lattice_site) + 1) / 2); // Green
                data[i + 2] = 128;  // Blue (constant)
                data[i + 3] = 255;  // Alpha (fully opaque)

                if (lattice.vortices !== null) {
                    let image_scaling = 1;
                    if (lattice.max_vortex > 0) {
                        image_scaling *= lattice.max_vortex * 2;
                    }
                    vortex_data[i] = (255 / image_scaling) * (lattice.vortices[x][y] + lattice.max_vortex);
                    vortex_data[i + 1] = 0;
                    vortex_data[i + 2] = 0;
                    vortex_data[i + 3] = 255;
                }
            }
        }
    
        ctx.putImageData(imageData, 0, 0);
        vortex_ctx.putImageData(vortexImageData, 0, 0);
    
        requestAnimationFrame(updateAndDraw);
    }
    
    requestAnimationFrame(updateAndDraw);
}

let lattice = null;

function fetch_parameters() {
    const parameters = {};
    for (const param of ["dimension", "o_dimension", "N"]) {
        parameters[param] = parseInt(document.getElementById(param).value);
    }

    parameters.beta = 1 / parseFloat(document.getElementById("temperature").value);

    return parameters;
}

function update_parameters() {
    const parameters = fetch_parameters();

    if (lattice === null) {
        lattice = new Lattice(parameters.dimension, parameters.N, parameters.beta, parameters.o_dimension);
    }
    
    if (parameters.beta !== lattice.beta) {
        lattice.beta = parameters.beta;
    }

    let lattice_change = false;
    for (const param of ['o_dimension', 'dimension', 'N']) {
        if (parameters[param] !== lattice[param]) {
            lattice_change = true;
            break;
        }
    }
    if (lattice_change) {
        lattice.set_lattice_params(parameters.dimension, parameters.N, parameters.o_dimension);
    }
    

    document.getElementById("temperature_display").textContent = (1 / parameters.beta).toString();
}


