import numpy as np
import numpy.random as rn
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from time import process_time
from functions import *




Temp = 0.1
J = 1         # coupling constant
L = 20        # system size

concentration_0 = 1
concentration_max = 0.05
concentration_step_size = -0.05

therm_steps = 1000
mc_steps = 1000
skip_step = 1
simulation_step_no = 1


concentration_array = np.arange(concentration_0, concentration_max + concentration_step_size, concentration_step_size)
concentration_array = np.round(concentration_array, 3)

concentration_step_no = len(concentration_array)
print(f"Total number of steps: {concentration_step_no}")
print()

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
        thermalized_state = ising_mc_thermalization(random_state, Temp, L, therm_steps, J)
        av_energy_per_spin_arr[j], av_magnetization_per_spin_arr[j], av_susceptibility_per_spin_arr[j] = ising_mc_simulation(thermalized_state, Temp, L, mc_steps, J, skip_step = skip_step)

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
plt.title(f"Magnetization per Particle vs Concentration \nSystem Size (L): {L}, Interaction (J): {J}, Themperature: {Temp} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Concentration (x)")
plt.ylabel("Magnetization per Particle")
plt.grid()
plt.show()


plt.plot(concentration_array, np.abs(energy_v_concentration_arr))
plt.title(f"Energy per Particle vs Concentration \nSystem Size (L): {L}, Interaction (J): {J}, Themperature: {Temp} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Concentration (x)")
plt.ylabel("Energy per Particle")
plt.grid()
plt.show()