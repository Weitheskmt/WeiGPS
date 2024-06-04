import numpy as np
import matplotlib.pyplot as plt


omega = 1
t = 1
psi = 1.5
A = 1
dt = 0.01

H0 = np.array([[0, 0],[0,2]])
N1 = np.array([[0,1],[1,0]])
Iden = np.array([[1,0],[0,1]])

N = 5

HF = np.zeros([2*(2*N+1),2*(2*N+1)],dtype='complex128')

for i in range(-N,N+1):
    print(i)
    for j in range(2):
        for k in range(2):
            HF[(i+N)*2+j,(i+N)*2+k] = H0[j,k] + i*omega*Iden[j,k]

print(HF)


for i in range(-N,N):
    for j in range(2):
        for k in range(2):
            HF[(i+N)*2+j,(i+N+1)*2+k] = A*N1[j,k]*np.exp(-1j*psi)/2 
            HF[(i+N+1)*2+j,(i+N)*2+k] = A*N1[j,k]*np.exp(1j*psi)/2 

def RK4(H0,N1, t, wavefun,dt):
    
    V12 = A* np.cos(omega*t + psi)

    wave_01 = -1j*np.matmul(H0+N1*V12,wavefun) 

    V12 = A* np.cos(omega*(t+dt/2.0) + psi)

    wave_02 = -1j*np.matmul(H0+N1*V12,wavefun+wave_01*dt/2.0) 

    V12 = A* np.cos(omega*(t+dt/2.0) + psi)

    wave_03 = -1j*np.matmul(H0+N1*V12,wavefun+wave_02*dt/2.0) 

    V12 = A* np.cos(omega*(t+dt) + psi)

    wave_04 = -1j*np.matmul(H0+N1*V12,wavefun+wave_03*dt) 


    wavefun = wavefun + dt*(wave_01 + 2.0*wave_02 + 2.0*wave_03 + wave_04)/6.0

    return wavefun

def RK4_HF(HF,t, wavefun,dt):
    
    wave_01 = -1j*np.matmul(HF,wavefun) 

    wave_02 = -1j*np.matmul(HF,wavefun+wave_01*dt/2.0) 

    wave_03 = -1j*np.matmul(HF,wavefun+wave_02*dt/2.0) 

    wave_04 = -1j*np.matmul(HF,wavefun+wave_03*dt) 


    wavefun = wavefun + dt*(wave_01 + 2.0*wave_02 + 2.0*wave_03 + wave_04)/6.0

    return wavefun


# wavefun_0 = np.array([0,1])
# wavefun_F_0 = np.zeros([2*(2*N+1)])

wavefun = np.array([1,0])
wavefun_F = np.zeros([2*(2*N+1)])
for i in range(-N,N+1):
    for j in range(2):
        wavefun_F[(i+N)*2+j]= wavefun[j]

print(wavefun_F)

n_step = 10000
t_list = np.linspace(0,dt*n_step,n_step)
y_list = np.linspace(0,dt*n_step,n_step)
z_list = np.zeros(n_step)

for it in range(n_step):
    t = it*dt
    wavefun = RK4(H0,N1,t,wavefun,dt)
    wavefun_F = RK4_HF(HF,t,wavefun_F,dt)
    flag=1
    y_list[it] = wavefun[flag] 
    for i in range(-N, N+1):
        z_list[it] += wavefun_F[(i+N)*2+flag]*np.exp(1j*i*omega*t)

plt.plot(t_list,y_list)
plt.plot(t_list,z_list/11)
plt.xlim(0,10)
plt.show()

print(HF)

