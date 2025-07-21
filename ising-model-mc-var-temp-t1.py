import numpy as np
import numpy.random as rn
from numba import jit, njit, prange
import matplotlib.pyplot as plt
from time import process_time
from functions import *




concentration = 0.3
J = 1         # coupling constant
L = 20        # system size

temp_0 = 6
temp_max = 0.2
temp_step_size = -0.2

therm_steps = 1000
mc_steps = 1000
skip_step = 1
simulation_step_no = 1


temp_array = np.arange(temp_0, temp_max + temp_step_size, temp_step_size)
temp_array = np.round(temp_array, 3)

temp_step_no = len(temp_array)
print(f"Total number of steps: {temp_step_no}")
print()

energy_v_temp_arr = np.zeros((temp_step_no))
magnetization_v_temp_arr = np.zeros((temp_step_no))
susceptibility_v_temp_arr = np.zeros((temp_step_no))



for i, temp in enumerate(temp_array):
    start_time = process_time()

    av_energy_per_spin_arr = np.zeros((simulation_step_no))
    av_magnetization_per_spin_arr = np.zeros((simulation_step_no))
    av_susceptibility_per_spin_arr = np.zeros((simulation_step_no))

    for j in range(simulation_step_no):
        # random_state = random_initial_state_generator(L, concentration)
        random_state = all_up_initial_state_generator(L, concentration)
        thermalized_state = ising_mc_thermalization(random_state, temp, L, therm_steps, J)
        av_energy_per_spin_arr[j], av_magnetization_per_spin_arr[j], av_susceptibility_per_spin_arr[j] = ising_mc_simulation(thermalized_state, temp, L, mc_steps, J, skip_step = skip_step)

    end_time = process_time()

    energy_v_temp_arr[i] = av_energy_per_spin_arr.mean()
    magnetization_v_temp_arr[i] = av_magnetization_per_spin_arr.mean()
    susceptibility_v_temp_arr[i] = av_susceptibility_per_spin_arr.mean()


    print(f"Concentration Step: {i + 1}")
    print(f"Concentration: {concentration}")
    print(f"Temperature: {temp}")
    # print(av_energy_per_spin, '\n', av_magnetization_per_spin, '\n', av_susceptibility_per_spin)
    print(f"Average Energy per Spin: {energy_v_temp_arr[i]}")
    print(f"Average Magnetization per Spin: {magnetization_v_temp_arr[i]}")
    # print(f"Average Susceptibility per Spin: {susceptibility_v_concentration_arr[i]}")
    print(f"CPU time taken: {end_time - start_time}")
    print("-------------------------------------------------------------------")


specific_heat_v_temp_arr = central_difference_derivative(temp_array, energy_v_temp_arr)



plt.plot(temp_array, np.abs(magnetization_v_temp_arr))
plt.title(f"Magnetization per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J}, Concentration: {concentration} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization per Particle")
plt.grid()
plt.show()


plt.plot(temp_array, np.abs(energy_v_temp_arr))
plt.title(f"Energy per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J}, Concentration: {concentration} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Temperature (T)")
plt.ylabel("Energy per Particle")
plt.grid()
plt.show()


plt.plot(temp_array[1: -1], specific_heat_v_temp_arr)
plt.title(f"Specific Heat per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J}, Concentration: {concentration} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Temperature (T)")
plt.ylabel("Specific Heat per Particle")
plt.grid()
plt.show()


plt.plot(temp_array, susceptibility_v_temp_arr)
plt.title(f"Susceptibility per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J}, Concentration: {concentration} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Temperature (T)")
plt.ylabel("Susceptibility per Particle")
plt.grid()
plt.show()