import numpy as np
# import numpy.random as rn
# from numba import jit, njit, prange
# import matplotlib.pyplot as plt
from time import process_time, time
from functions import *
# from matplotlib.ticker import AutoMinorLocator, MultipleLocator




tot_start_time = time()
concentration_array = np.round(np.linspace(0.1, 1, 10), 3)
concentration_step_no = len(concentration_array)
J = 1         # coupling constant
L = 20        # system size

temp_0 = 6
temp_max = 0.2
temp_step_size = -0.2

therm_steps = 1
mc_steps = 1
skip_step = 1
simulation_step_no = 1


temp_array = np.arange(temp_0, temp_max + temp_step_size, temp_step_size)
temp_array = np.round(temp_array, 3)

temp_step_no = len(temp_array)
print(f"Total number of steps: {temp_step_no}")
print()

energy_v_temp_arr = np.zeros((concentration_step_no, temp_step_no))
magnetization_v_temp_arr = np.zeros((concentration_step_no, temp_step_no))
susceptibility_v_temp_arr = np.zeros((concentration_step_no, temp_step_no))

for k, concentration in enumerate(concentration_array):

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

        energy_v_temp_arr[k, i] = av_energy_per_spin_arr.mean()
        magnetization_v_temp_arr[k, i] = av_magnetization_per_spin_arr.mean()
        susceptibility_v_temp_arr[k, i] = av_susceptibility_per_spin_arr.mean()


        print(f"Concentration Step: {i + 1}")
        print(f"Concentration: {concentration}")
        print(f"Temperature: {temp}")
        # print(av_energy_per_spin, '\n', av_magnetization_per_spin, '\n', av_susceptibility_per_spin)
        # print(f"Average Energy per Spin: {energy_v_temp_arr[i]}")
        # print(f"Average Magnetization per Spin: {magnetization_v_temp_arr[i]}")
        # print(f"Average Susceptibility per Spin: {susceptibility_v_concentration_arr[i]}")
        print(f"CPU time taken: {end_time - start_time}")
        print("-------------------------------------------------------------------")


    # specific_heat_v_temp_arr[] = central_difference_derivative(temp_array, energy_v_temp_arr)

specific_heat_v_temp_arr = []
for step in range(concentration_step_no):
    specific_heat_v_temp_arr.append(central_difference_derivative(temp_array, energy_v_temp_arr[step]))
specific_heat_v_temp_arr = np.array(specific_heat_v_temp_arr)

print('------------------------------------------------------------')

print(concentration_array)
print(temp_array)
print(magnetization_v_temp_arr)
print(specific_heat_v_temp_arr)
print(susceptibility_v_temp_arr)
print(energy_v_temp_arr)

print(np.shape(concentration_array))
print(np.shape(temp_array))
print(np.shape(magnetization_v_temp_arr))
print(np.shape(specific_heat_v_temp_arr))
print(np.shape(susceptibility_v_temp_arr))
print(np.shape(energy_v_temp_arr))

np.save("concentration_array.npy", concentration_array)
np.save("temp_array.npy", temp_array)
np.save("magnetization_v_temp_arr.npy",magnetization_v_temp_arr)
np.save("energy_v_temp_arr.npy",energy_v_temp_arr)
np.save("specific_heat_v_temp_arr.npy",specific_heat_v_temp_arr)
np.save("susceptibility_v_temp_arr.npy",susceptibility_v_temp_arr)


print(f"Totatl Time Taken: {time() - tot_start_time}")


# for step in range(concentration_step_no):
#     plt.plot(temp_array, np.abs(magnetization_v_temp_arr[step]), label= f'x: {concentration_array[step]}')
# plt.title(f"Magnetization per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J}\nMonte Carlo Step: {mc_steps}")
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
# plt.tick_params(axis='both', which='minor', length=4, width=0.8, direction='out')
# plt.tick_params(which='major',length=8,width=1.5, direction='inout',labelsize=12)
# plt.xlim((0, 6))
# plt.grid()
# plt.grid(visible=True, which='minor', axis='both', linestyle='--')
# plt.xlabel("Temperature (T)")
# plt.ylabel("Magnetization per Particle")
# plt.legend()
# plt.show()


# for step in range(concentration_step_no):
#     plt.plot(temp_array, np.abs(energy_v_temp_arr[step]), label= f'x: {concentration_array[step]}')
# plt.title(f"Energy per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J} \nMonte Carlo Step: {mc_steps}")
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
# plt.tick_params(axis='both', which='minor', length=4, width=0.8, direction='out')
# plt.tick_params(which='major',length=8,width=1.5, direction='inout',labelsize=12)
# plt.xlim((0, 6))
# plt.grid()
# plt.grid(visible=True, which='minor', axis='both', linestyle='--')
# plt.xlabel("Temperature (T)")
# plt.ylabel("Energy per Particle")
# plt.legend()
# plt.grid()
# plt.show()


# for step in range(concentration_step_no):
#     specific_heat_v_temp_arr = central_difference_derivative(temp_array, energy_v_temp_arr[step])
#     plt.plot(temp_array[1: -1], np.abs(specific_heat_v_temp_arr), label= f'x: {concentration_array[step]}')
# plt.title(f"Specific Heat per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J} \nMonte Carlo Step: {mc_steps}")
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
# plt.tick_params(axis='both', which='minor', length=4, width=0.8, direction='out')
# plt.tick_params(which='major',length=8,width=1.5, direction='inout',labelsize=12)
# plt.xlim((0, 6))
# plt.grid()
# plt.grid(visible=True, which='minor', axis='both', linestyle='--')
# plt.xlabel("Temperature (T)")
# plt.ylabel("Specific Heat per Particle")
# plt.grid()
# plt.show()


# for step in range(concentration_step_no):
#     plt.plot(temp_array, np.abs(susceptibility_v_temp_arr[step]), label= f'x: {concentration_array[step]}')
# plt.title(f"Susceptibility per Particle vs Themperature \nSystem Size (L): {L}, Interaction (J): {J} \nMonte Carlo Step: {mc_steps}")
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
# plt.tick_params(axis='both', which='minor', length=4, width=0.8, direction='out')
# plt.tick_params(which='major',length=8,width=1.5, direction='inout',labelsize=12)
# plt.xlim((0, 6))
# plt.grid()
# plt.grid(visible=True, which='minor', axis='both', linestyle='--')
# plt.xlabel("Temperature (T)")
# plt.ylabel("Susceptibility per Particle")
# plt.legend()
# plt.grid()
# plt.show()