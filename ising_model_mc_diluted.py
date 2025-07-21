import numpy as np
import numpy.random as rn
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from time import process_time




# @njit
# @jit(nopython=True, parallel=True)
def random_initial_state_generator(L, concentration): # Checked OK
    state = np.zeros((L, L, L))
    index_list = []


    for i in prange(L):
        for j in prange(L):
            for k in prange(L):
                if rn.rand() <= 0.5:
                    state[i, j, k] = -1
                else:
                    state[i, j, k] = 1

    for i in prange(L):
        for j in prange(L):
            for k in prange(L):
                index_list.append([i, j, k])

    rn.shuffle(index_list)

    for i in prange(int(round(((1 - concentration) * L * L * L), 5))):
        coordinate = index_list.pop(0)
        state[coordinate[0], coordinate[1], coordinate[2]] = 0
    
    return state



def all_up_initial_state_generator(L, concentration): # Checked OK
    state = np.ones((L, L, L))
    index_list = []


    for i in prange(L):
        for j in prange(L):
            for k in prange(L):
                index_list.append([i, j, k])

    rn.shuffle(index_list)

    for i in prange(int(round(((1 - concentration) * L * L * L), 5))):
        coordinate = index_list.pop(0)
        state[coordinate[0], coordinate[1], coordinate[2]] = 0
    
    return state



# @jit(nopython=True, parallel=True)
@njit
def initial_state_test(L, state):
    test_conc = 0

    for i in prange(L):
        for j in prange(L):
            for k in prange(L):
                if state[i, j, k] == 0:
                    test_conc += 1
    
    return (test_conc / (L * L * L))



# @jit(nopython=True, parallel=True)
@njit
def calculate_magnetization_per_spin(state, L): # Checked OK
  magnetization = 0

  for i in prange(L):
    for j in prange(L):
      for k in prange(L):
        magnetization += state[i, j, k]

  return magnetization / (L**3)



# @jit(nopython=True, parallel=True)
@njit
def calculate_energy_per_spin(state, L, J, B=0): # Checked OK
    energy_J = 0
    energy_B = 0

    for i in prange(L):
        for j in prange(L):
            for k in prange(L):
                i_prev = (i - 1) % L
                i_next = (i + 1) % L
                j_prev = (j - 1) % L
                j_next = (j + 1) % L
                k_prev = (k - 1) % L
                k_next = (k + 1) % L
                
                energy_J += -J * state[i, j, k] * (state[i_prev, j, k] + state[i_next, j, k] + state[i, j_prev, k] + state[i, j_next, k] + state[i, j, k_prev] + state[i, j, k_next])
                energy_B += -B * state[i, j, k]

    return (energy_J + energy_B) / (2 * L**3)



# @jit(nopython=True, parallel=True)
@njit
def next_state_generator(state, L):
    site = np.array([rn.randint(0, L - 1), rn.randint(0, L - 1), rn.randint(0, L - 1)])
    site_spin = state[site[0], site[1], site[2]]

    if site_spin == 1:
        state[site[0], site[1], site[2]] = -1
    elif site_spin == -1:
        state[site[0], site[1], site[2]] = 1

    return state, site



# @jit(nopython=True)
@njit
def change_in_energy(current_state, next_state, site, L, J, B=0):
    i = site[0]
    j = site[1]
    k = site[2]

    i_prev = (i - 1) % L
    i_next = (i + 1) % L
    j_prev = (j - 1) % L
    j_next = (j + 1) % L
    k_prev = (k - 1) % L
    k_next = (k + 1) % L

    current_state_site_energy = -J * current_state[i, j, k] * (current_state[i_prev, j, k] + current_state[i_next, j, k] + current_state[i, j_prev, k] + current_state[i, j_next, k] + current_state[i, j, k_prev] + current_state[i, j, k_next])

    next_state_site_energy = -J * next_state[i, j, k] * (next_state[i_prev, j, k] + next_state[i_next, j, k] + next_state[i, j_prev, k] + next_state[i, j_next, k] + next_state[i, j, k_prev] + next_state[i, j, k_next])

    return next_state_site_energy - current_state_site_energy



# @jit(nopython=True, parallel=True)
@njit
def central_difference_derivative(x_array, y_array):
    N = len(x_array)
    cnt_der = np.zeros((N - 2))

    for i in prange(1, N-1):
        cnt_der[i - 1] = (y_array[i + 1] - y_array[i - 1]) / (x_array [i + 1] - x_array[i - 1])
    
    return cnt_der



# @jit(nopython=True, parallel=True)
@njit
def ising_mc_thermalization(random_state, T, L, therm_steps, J):
    state = random_state.copy()

    for l in prange(therm_steps):
        for s in prange(L * L * L):
            next_state = state.copy()

            i = rn.randint(L)
            j = rn.randint(L)
            k = rn.randint(L)

            if next_state[i, j , k] == 0:
                continue

            next_state[i, j , k] = -1 * state[i, j, k]
            energy_change = change_in_energy(state, next_state, [i, j, k], L, J)

            P_acc = np.exp(-1 * energy_change / T)

            if energy_change < 0:
                state = next_state.copy()
            
            elif rn.rand() <= P_acc:
                state = next_state.copy()

    return state



@njit
def ising_mc_simulation(thermalized_state, T, L, mc_steps, J, skip_step = 10):
    state = thermalized_state.copy()

    energy_per_spin = 0
    magnetization_per_spin_arr = np.zeros(((mc_steps // skip_step) + 1))

    for l in prange(mc_steps):
        for s in prange(L * L * L):
            next_state = state.copy()

            i = rn.randint(L)
            j = rn.randint(L)
            k = rn.randint(L)

            if next_state[i, j , k] == 0:
                continue

            next_state[i, j , k] = -1 * state[i, j, k]
            energy_change = change_in_energy(state, next_state, [i, j, k], L, J)

            P_acc = np.exp(-1 * energy_change / T)

            if energy_change < 0:
                state = next_state.copy()
            
            elif rn.rand() <= P_acc:
                state = next_state.copy()
    
        if (l + 1) % skip_step == 0:
            energy_per_spin += calculate_energy_per_spin(state, L, J, B=0)
            magnetization_per_spin_arr[int((l + 1) / skip_step)] = calculate_magnetization_per_spin(state, L)
    
    magnetization_per_spin = magnetization_per_spin_arr.mean()
    
    susceptibility_per_spin = np.abs(((np.square(magnetization_per_spin_arr)).mean() - (magnetization_per_spin**2)) / T)
    
    return ((energy_per_spin * skip_step) / mc_steps), magnetization_per_spin, susceptibility_per_spin




T = 0.1
J = 1
L = 20

concentration_0 = 1
concentration_max = 0.02
concentration_step_size = -0.02

therm_steps = 1000
mc_steps = 1000
skip_step = 1
simulation_step_no = 1

concentration_array = np.arange(concentration_0, concentration_max + concentration_step_size, concentration_step_size)
concentration_array = np.round(concentration_array, 3)

concentration_step_no = len(concentration_array)
print(concentration_step_no)

energy_v_concentration_arr = np.zeros((concentration_step_no))
magnetization_v_concentration_arr = np.zeros((concentration_step_no))


for i, concentration in enumerate(concentration_array):
    start_time = process_time()

    av_energy_per_spin_arr = np.zeros((simulation_step_no))
    av_magnetization_per_spin_arr = np.zeros((simulation_step_no))
    av_susceptibility_per_spin_arr = np.zeros((simulation_step_no))

    for j in range(simulation_step_no):
        random_state = random_initial_state_generator(L, concentration)
        # random_state = all_up_initial_state_generator(L, concentration)
        thermalized_state = ising_mc_thermalization(random_state, T, L, therm_steps, J)
        av_energy_per_spin_arr[j], av_magnetization_per_spin_arr[j], av_susceptibility_per_spin_arr[j] = ising_mc_simulation(thermalized_state, T, L, mc_steps, J, skip_step = skip_step)

    end_time = process_time()

    energy_v_concentration_arr[i] = av_energy_per_spin_arr.mean()
    magnetization_v_concentration_arr[i] = av_magnetization_per_spin_arr.mean()


    print(f"Concentration Step: {i + 1}")
    print(f"Concentration: {concentration, initial_state_test(L, random_state)}")
    # print(av_energy_per_spin, '\n', av_magnetization_per_spin, '\n', av_susceptibility_per_spin)
    print(f"Average Energy per Spin: {energy_v_concentration_arr[i]}")
    print(f"Average Magnetization per Spin: {magnetization_v_concentration_arr[i]}")
    # print(f"Average Susceptibility per Spin: {susceptibility_v_concentration_arr[i]}")
    print(f"CPU time taken: {end_time - start_time}")
    print("-------------------------------------------------------------------")



plt.plot(concentration_array, np.abs(magnetization_v_concentration_arr))
plt.title(f"Magnetization per Particle vs Concentration \nSystem Size (L): {L}, Interaction (J): {J}, Themperature: {T} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Concentration (x)")
plt.ylabel("Magnetization per Particle")
plt.grid()
plt.show()


plt.plot(concentration_array, np.abs(energy_v_concentration_arr))
plt.title(f"Energy per Particle vs Concentration \nSystem Size (L): {L}, Interaction (J): {J}, Themperature: {T} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Concentration (x)")
plt.ylabel("Energy per Particle")
plt.grid()
plt.show()
