# -*- coding: utf-8 -*-

from weigps import WeiGPS
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import sys

OMEGA = VALUE_OMEGA
DISORDER = VALUE_DISORDER
UP = VALUE_UP
SEED = VALUE_S

light_size = 1
molecule_size = 1000000
disorder = DISORDER # std_dev
frequency_molecule = list(WeiGPS.get_distribution(mean=0,num_samples=molecule_size,disorder=disorder,random_seed=int(10),cut_off=[-7,7]))
frequency_light = list(WeiGPS.get_distribution(mean=np.mean(frequency_molecule),num_samples=light_size,disorder=0))
hbar = 1
dt = 0.001
tf = 2000
Floquet_level = 3



phase_random = SEED
Omega = OMEGA
V_coupling_strength = 0.01

gamma = 0.1
phase_up = VALUE_UP
phase_dw = 32

phase = list(np.random.uniform(-1, 1, molecule_size)*(phase_up*np.pi/phase_dw))

w_left_lim = -20
w_right_lim = 20
w_length_lim = 1000

data = {
    "light_size": light_size, # the number of photon modes
    "molecule_size": molecule_size, # the number of molecule vibrational modes
    "hbar": hbar, # default 1

    "frequency_molecule": frequency_molecule, # vibrational frequency
    "frequency_light": frequency_light, # [frequency_light]
    "V_coupling_strength":V_coupling_strength, #1, 1.5, 2
    
    "dt": dt, # correlation function dt
    "tf": tf, # correlation function tf
    "Floquet_level": Floquet_level, # Floquet level
    "disorder": disorder, # std_dev

    "gamma":gamma, # another gaussian width (std_dev)

    "phase_random": phase_random, # random SEED, label random phase; if not random phase no need set None;
    "phase":phase, # phase: cos(\Omega*t+phase)
    "phase_up": phase_up, # range of 1/32 * [-np.pi, np.pi]
    "phase_dw": phase_dw, # range of 1/32 * [-np.pi, np.pi]
    "Omega": Omega, # w: cos(\Omega*t+phase)  0 0.1 0.5
    
    "w_left_lim": w_left_lim, # spectrum left end
    "w_right_lim": w_right_lim, # spectrum right end
    "w_length_lim": w_length_lim, # spectrum the number of w
}


json_file_path = f"data_{UP}.json"

with open(json_file_path, "w") as json_file:
    json.dump(data, json_file)

### 记录开始时间
start_time = time.time()

GPS = WeiGPS(json_file_path)

TIME, observable = GPS.run_dynamics()

GPS.plot_corr(TIME, observable,SAVE=True,PLOT=True)

# 记录结束时间
end_time = time.time()

# 计算经过的时间
elapsed_time = end_time - start_time

print(f"动力学经过的时间：{elapsed_time}秒")

