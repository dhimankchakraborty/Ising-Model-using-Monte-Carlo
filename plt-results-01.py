import numpy as np
import matplotlib.pyplot as plt



concentration_array = np.load("concentration_array_10.npy")
temp_array = np.load("temp_array_10.npy")
magnetization_v_temp_arr = np.load("magnetization_v_temp_arr_10.npy")
energy_v_temp_arr = np.load("energy_v_temp_arr_10.npy")
specific_heat_v_temp_arr = np.load("specific_heat_v_temp_arr_10.npy")
susceptibility_v_temp_arr = np.load("susceptibility_v_temp_arr_10.npy")


plt.tight_layout()
# print(magnetization_v_temp_arr)

for step in range(len(concentration_array)):
    plt.plot(temp_array, np.abs(magnetization_v_temp_arr[step]), label= f'x: {concentration_array[step]}')
plt.title(f"Magnetization per Particle vs Temperature \nSystem Size (L): {10}, Interaction (J): {1} \nMonte Carlo Step: {1000}")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization per Particle")
plt.legend()
plt.grid()
plt.savefig('plots\\a1.png', bbox_inches='tight', dpi=150)
# plt.show()
plt.clf()

for step in range(len(temp_array)):
    plt.plot(concentration_array, np.abs(magnetization_v_temp_arr[:,step]), label= f'T: {temp_array[step]}')
plt.title(f"Magnetization per Particle vs Concentration \nSystem Size (L): {10}, Interaction (J): {1} \nMonte Carlo Step: {1000}")
plt.xlabel("Concentration (x)")
plt.ylabel("Magnetization per Particle")
plt.legend()
plt.grid()
plt.savefig('plots\\a2.png', bbox_inches='tight', dpi=150)
# plt.show()
plt.clf()