import numpy as np
import matplotlib.pyplot as plt



concentration_array = np.load(f'concentration_array_0.1.npy')
magnetization_v_concentration_arr = np.load(f'abs_magnetization_v_concentration_arr_0.1.npy')
energy_v_concentration_arr = np.load(f'abs_energy_v_concentration_arr_0.1.npy')



plt.plot(concentration_array, np.abs(magnetization_v_concentration_arr))
# plt.title(f"Magnetization per Particle vs Concentration \nSystem Size (L): {L}, Interaction (J): {J}, Themperature: {T} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Concentration (x)")
plt.ylabel("Magnetization per Particle")
plt.grid()
plt.show()


plt.plot(concentration_array, np.abs(energy_v_concentration_arr))
# plt.title(f"Energy per Particle vs Concentration \nSystem Size (L): {L}, Interaction (J): {J}, Themperature: {T} \nMonte Carlo Step: {mc_steps}")
plt.xlabel("Concentration (x)")
plt.ylabel("Energy per Particle")
plt.grid()
plt.show()