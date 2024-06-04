# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix
from numba import njit
import time
from numba import jit
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
import json
import os
import warnings
warnings.filterwarnings("ignore")

class WeiGPS:
    def __init__(self, json_file_path):
        # 读取 JSON 文件
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # 检查 omega 是否为0

        # 从 data 字典中获取变量的值
        self.light_size = data["light_size"]
        self.total_size = data["total_size"]
        self.molecule_size = self.total_size-self.light_size

        self.molecule_start = data["molecule_start"]
        self.DeltaE = data["DeltaE"]
        self.scale = data["scale"]

        if data["V_coupling_strength"] is not None:
            self.V_coupling_strength = data["V_coupling_strength"]
        else:
            self.V_coupling_strength = self.DeltaE/(np.sqrt(self.total_size))/self.scale*500

        print("V_coupling_strength:", self.V_coupling_strength)

        self.hbar = data["hbar"]
        self.dt = data["dt"]
        if data.get('omega', 0) == 0:
            raise ValueError("Omega cannot be equal to 0, please provide an appropriate value.")
        else:
            self.omega = data["omega"] # not zero
        
        self.gamma = data["gamma"] #gaussian

        self.tf = data["tf"]
        self.steps = int(self.tf/self.dt)
        self.floquet_level = data["floquet_level"]
        self.disorder = data["disorder"]

        if data["e_molecule"] is None:
            self.diagonal_H11 = np.linspace(self.molecule_start,self.molecule_start+self.DeltaE,self.molecule_size)
        else:
            self.e_molecule = data["e_molecule"]
            if data["e_molecule_distribution"]:
                if data["seed"]:
                    np.random.seed(10)
                if self.disorder == 4:
                    # 定义边界 dis=2:[-7,7];dis=3:[-10,10];dis=4:[-12,12];
                    lower_bound = -13
                    upper_bound = 13
                elif self.disorder == 3:
                    lower_bound = -10
                    upper_bound = 10
                
                elif self.disorder == 2:
                    lower_bound = -7
                    upper_bound = 7

                elif self.disorder == 1:
                    lower_bound = -4
                    upper_bound = 4

                else:
                    lower_bound = data["left_lim"]
                    upper_bound = data["right_lim"]

                # 生成高斯分布
                mean = self.e_molecule  # 均值
                std_dev = self.disorder  # 标准差
                num_samples = self.molecule_size  # 样本数

                # 初始化样本列表
                samples = []
                # 生成满足条件的样本数
                while len(samples) < num_samples:
                    new_sample = np.random.normal(mean, std_dev)
                    if lower_bound <= new_sample <= upper_bound:
                        samples.append(new_sample)

                self.diagonal_H11 = np.array(samples)
                print(r"$\Delta \ E=$"+'{0}'.format(np.max(self.diagonal_H11)-np.min(self.diagonal_H11)))

        E0 = np.mean(self.diagonal_H11)
        if data["e_light"] is None:
            self.e_light = E0
        else:
            self.e_light = data["e_light"]

        print("e_light:",self.e_light)

        self.n_superblock = self.total_size * (2*self.floquet_level+1)

        if data["e_light_distribution"]:
            self.diagonal_H00 = np.random.normal(self.e_light, 1, self.light_size)
        else:
            self.diagonal_H00 = np.random.normal(self.e_light, 0, self.light_size)  # 生成符合均匀分布的随机值
            
        self.Diagonal = np.concatenate([self.diagonal_H00,self.diagonal_H11], axis=0)

        self.Pphoton = np.zeros(self.n_superblock)
        for q in range(-self.floquet_level, self.floquet_level+1):
            self.Pphoton[(self.total_size*(q+self.floquet_level+1)-self.total_size):(self.total_size*(q+self.floquet_level+1)-self.total_size)+self.light_size] = 1

        # Pmolecule = np.ones(n_superblock) - Pphoton
        self.phase_dw = data['phase_dw']
        self.phase_up = data['phase_up']
        self.phase_random = data['phase_random']
        self.phase_pi = data['phase']
        if int(self.phase_pi) == 0:
            self.phase_up = 0
        np.random.seed(self.phase_random)
        self.phase = np.random.uniform(-1, 1, self.molecule_size)*(self.phase_up*self.phase_pi/self.phase_dw)

        self.left_lim = data["left_lim"]
        self.right_lim = data["right_lim"]
        self.length_lim = data["length_lim"]

### coo algorithm
    @staticmethod
    @njit(parallel=True)
    def sparse_matrix_multiply(coo_matrix_a, coo_matrix_b):
        data_a, (row_a, col_a), (n_a, m_a) = coo_matrix_a
        data_b, (row_b, col_b), (n_b, m_b) = coo_matrix_b

        if m_a != n_b:
            raise ValueError("Incompatible matrix dimensions for multiplication")


        result_data = []
        result_row = []
        result_col = []

        for i in range(len(data_a)):
            for j in range(len(data_b)):
                if col_a[i] == row_b[j]:
                    row_idx_result = row_a[i]
                    col_idx_result = col_b[j]

                    # 寻找结果数组中是否已经存在相同的行列索引
                    found = False
                    for k in range(len(result_data)):
                        if result_row[k] == row_idx_result and result_col[k] == col_idx_result:
                            result_data[k] += data_a[i] * data_b[j]
                            found = True
                            break
                    # 如果不存在，则添加新的元素
                    if not found:
                        result_data.append(data_a[i] * data_b[j])
                        result_row.append(row_idx_result)
                        result_col.append(col_idx_result)

        return np.array(result_data), (np.array(result_row), np.array(result_col)), (n_a, m_b)
    

    @staticmethod
    @njit(parallel=True)
    def transpose_coo(coo_matrix_input, conj=True):
        data, (row, col), (n ,m) = coo_matrix_input
        if conj:
            return np.conj(data), (col, row), (m ,n)
        else:
            return data, (col, row), (m ,n) 
    
    @staticmethod
    @njit(parallel=True)
    def create_coo_matrix_nonzero(arr, shape,format_type='row'):
        """
        Create a COO format sparse matrix from a 1D array based on the specified format.

        Parameters:
        - arr: 1D array
        - format_type: str, optional
            Specify the format to construct the sparse matrix. Options: 'row', 'col', 'diagonal'.
            Default is 'row'.

        Returns:
        - coo_matrix: scipy.sparse.coo_matrix
            COO format sparse matrix.
        """

        if format_type not in ['row', 'col', 'diagonal']:
            raise ValueError("Invalid format_type. Choose from 'row', 'col', or 'diagonal.")

        nonzero_indices = np.nonzero(arr)[0]

        if format_type == 'row':
            row_indices = np.zeros_like(nonzero_indices)
            col_indices = nonzero_indices
            
        elif format_type == 'col':
            row_indices = nonzero_indices
            col_indices = np.zeros_like(nonzero_indices)
            
        elif format_type == 'diagonal':
            row_indices = nonzero_indices
            col_indices = nonzero_indices
        return arr[nonzero_indices], (row_indices, col_indices), shape
    
    @staticmethod
    def dense(M):
        return coo_matrix((M[:2]),shape=M[2]).toarray()
    

    @staticmethod
    def save_array(data_folder, file_name, my_array):
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # 保存数组到文件
        np.save(os.path.join(data_folder, file_name), my_array)
    
### coo construct algorithm

    def V(self):
        indices = range(self.light_size)
        out = np.concatenate([np.ones(self.total_size-self.light_size)*self.V_coupling_strength*0.5 for _ in indices])
        return out

    def V_plus(self):
        indices = range(self.light_size)
        out = np.concatenate([np.ones(self.total_size-self.light_size)*self.V_coupling_strength*0.5*np.exp(1j*self.phase) for _ in indices])
        return out

    def V_minus(self):
        indices = range(self.light_size)
        out = np.concatenate([np.ones(self.total_size-self.light_size)*self.V_coupling_strength*0.5*np.exp(-1j*self.phase) for _ in indices])
        return out

    def row(self,n,i):
        return self.total_size*(n+self.floquet_level)+i

    def column(self,m,j):
        return self.total_size*(m+self.floquet_level)+j
    
### diagonal
    def diagonal(self):
        i_index=np.arange(self.total_size)
        row_diagonal = {}
        data_diagonal = {}
        for n in range(-self.floquet_level,self.floquet_level+1):
            row_diagonal['{0}'.format(n)] = self.row(n,i_index)
            data_diagonal['{0}'.format(n)] = self.Diagonal + n * self.hbar * self.omega

        indices = range(-self.floquet_level,self.floquet_level+1)  # -3 -> 3
        row_indice_diagonal = np.concatenate([row_diagonal[str(i)] for i in indices])
        data_indice_diagonal = np.concatenate([data_diagonal[str(i)] for i in indices])

        return row_indice_diagonal,data_indice_diagonal
    
### upper n=m+1
    def upper_nmp1(self):
        indices = range(self.light_size)  # 0 -> light_size
        i_index=np.ones(self.total_size)
        j_index=np.arange(self.total_size)
        i_index = np.concatenate([i_index[self.light_size:]*i for i in indices])
        j_index = np.concatenate([j_index[self.light_size:] for _ in indices])

        row_upper_mp1 = {}
        col_upper_mp1 = {}
        data_upper_mp1 = {}

        for n in range(-self.floquet_level,self.floquet_level+1):
            m=n-1
            if m+self.floquet_level >=0 :
                row_upper_mp1['{0}'.format(n)] = self.row(n,i_index)
                col_upper_mp1['{0}'.format(n)] = self.column(m,j_index)
                if int(self.phase_up)==0:
                    data_upper_mp1['{0}'.format(n)] = self.V() #V no phase
                else:
                    data_upper_mp1['{0}'.format(n)] = self.V_minus() #V_minus


        indices = range(-self.floquet_level+1,self.floquet_level+1)  # -2 -> 3
        row_indice_upper_mp1 = np.concatenate([row_upper_mp1[str(i)] for i in indices])
        col_indice_upper_mp1 = np.concatenate([col_upper_mp1[str(i)] for i in indices])
        data_indice_upper_mp1 = np.concatenate([data_upper_mp1[str(i)] for i in indices])
        return row_indice_upper_mp1, col_indice_upper_mp1, data_indice_upper_mp1
    
### upper n=m-1
    def upper_nmm1(self):

        indices = range(self.light_size)  # 0 -> light_size
        i_index=np.ones(self.total_size)
        j_index=np.arange(self.total_size)
        i_index = np.concatenate([i_index[self.light_size:]*i for i in indices])
        j_index = np.concatenate([j_index[self.light_size:] for _ in indices])

        row_upper_mm1 = {}
        col_upper_mm1 = {}
        data_upper_mm1 = {}

        for n in range(-self.floquet_level,self.floquet_level+1):
            m=n+1
            if m+self.floquet_level <=2*self.floquet_level:
                row_upper_mm1['{0}'.format(n)] = self.row(n,i_index)
                col_upper_mm1['{0}'.format(n)] = self.column(m,j_index)
                if int(self.phase_up)==0:
                    data_upper_mm1['{0}'.format(n)] = self.V() # V_plus
                else:
                    data_upper_mm1['{0}'.format(n)] = self.V_plus() # V_plus

        indices = range(-self.floquet_level,self.floquet_level) # -3 -> 2
        row_indice_upper_mm1 = np.concatenate([row_upper_mm1[str(i)] for i in indices])
        col_indice_upper_mm1 = np.concatenate([col_upper_mm1[str(i)] for i in indices])
        data_indice_upper_mm1 = np.concatenate([data_upper_mm1[str(i)] for i in indices])

        return row_indice_upper_mm1, col_indice_upper_mm1, data_indice_upper_mm1

### sparse_matrix

    def sparse_matrix(self):
        row_indice_diagonal,data_indice_diagonal = self.diagonal()
        row_indice_upper_mp1, col_indice_upper_mp1, data_indice_upper_mp1 = self.upper_nmp1()
        row_indice_upper_mm1, col_indice_upper_mm1, data_indice_upper_mm1 = self.upper_nmm1()

        row_indice = np.concatenate((row_indice_diagonal, row_indice_upper_mp1, row_indice_upper_mm1, col_indice_upper_mp1, col_indice_upper_mm1))
        col_indice = np.concatenate((row_indice_diagonal, col_indice_upper_mp1, col_indice_upper_mm1, row_indice_upper_mp1, row_indice_upper_mm1))
        data = np.concatenate([data_indice_diagonal, data_indice_upper_mp1, data_indice_upper_mm1, np.conj(data_indice_upper_mp1), np.conj(data_indice_upper_mm1)])

        
        return row_indice,col_indice,data

    def sparse_matrix_eigsh(self,k=30,SAVE=True):
        row_indice,col_indice,data = self.sparse_matrix()
        sparse_matrix = coo_matrix((data, (row_indice, col_indice)), shape=(self.n_superblock, self.n_superblock))

        if int(self.phase_up)==0:
            self.eigenvalues, self.eigenvectors = eigsh(sparse_matrix, k=k, which='BE')
        else:
            self.eigenvalues, self.eigenvectors = eigsh(sparse_matrix, k=k, which='LM')
        
        if SAVE:
            np.save("./data/data_evals_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).npy".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random), self.eigenvalues)
            # np.save("./data/data_evecs_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).npy".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random), self.eigenvectors)
            print("evals:", self.eigenvalues)
            
        return self.eigenvalues,self.eigenvectors

### spectrum
    def gaussian(self,w):

        out = np.exp(-(np.diag(self.eigenvalues)-w)**2/self.gamma**2)/(self.gamma*np.sqrt(np.pi))
        # 获取非零元素的坐标
        nonzero_coords = np.nonzero(out)
        M_gaussian = out[nonzero_coords], nonzero_coords, (out.shape)
        return M_gaussian

    def spectrum(self):
        w = np.linspace(self.left_lim,self.right_lim,self.length_lim)
        result = np.zeros(self.length_lim)
        PSI0 = self.create_coo_matrix_nonzero(self.Pphoton, format_type='row',shape=(1,self.n_superblock))
        PSI0_PSI0 = self.sparse_matrix_multiply(self.transpose_coo(PSI0),PSI0)
        # 获取非零元素的坐标
        nonzero_coords = np.nonzero(self.eigenvectors)
        M_evecs = self.eigenvectors[nonzero_coords], nonzero_coords, (self.eigenvectors.shape)

        for i in range(len(w)):
            A = self.sparse_matrix_multiply(PSI0_PSI0,M_evecs)
            B = self.sparse_matrix_multiply(self.transpose_coo(M_evecs),A)
            C = self.sparse_matrix_multiply(self.gaussian(w[i]),self.transpose_coo(B))
            result_vector_p1 = self.dense(C)

            result[i] = np.trace(result_vector_p1)


        near_zero_indices = np.where(np.abs(w) < 1)[0]

        # 删除 P 中对应索引的元素
        self.P_filtered = result
        self.w_filtered = w
        self.P_filtered[near_zero_indices] = 0

        return self.w_filtered, self.P_filtered
    
    def get_DOS(self,k):

        self.sparse_matrix_eigsh(k)
        self.spectrum()

        init = 0
        de = (self.w_filtered[1]-self.w_filtered[0])
        for i in range(len(self.w_filtered)):
            init+=self.P_filtered[i]*de
        print("init:", init)
        return self.w_filtered, self.P_filtered/init
    
### plot and save

    def plot_spectrum(self,w,p,SAVE=False,PLOT=False):

        if PLOT:

            # DMD = np.vstack((TIME,observable)).T
            plt.figure(dpi=350)

            plt.hist(self.diagonal_H11,bins=30,density=True)
            

            plt.axvline(x=self.e_light, color='green', alpha=0.5,lw=0.8,linestyle='--', label=r'$E_0$')

            plt.plot(w,np.abs(p),c='red',label=r'$\{<\psi_i|(\delta(\tilde{\lambda}-\omega) \tilde{U}^{\dagger} P_0 \tilde{U})|\psi_i>\}$')

            # plt.plot(w_filtered2, P_filtered2,'--',c='green',label=r'$Tr(\delta(\lambda-w) P_0 )$')
            plt.xlim(self.left_lim,self.right_lim)

            plt.xlabel(r'$\omega$')
            plt.ylabel('Spectra')

            plt.legend()

            plt.tight_layout()

            if not os.path.exists("./figs/"):
                os.makedirs("./figs/")
            plt.savefig('./figs/fig_Spectra_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).png'.format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random))
            plt.savefig("./figs/fig_e_molecule_distribution_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).png".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random))

        # plt.ylim(0,1)

        if SAVE:
            if not os.path.exists("./data/"):
                os.makedirs("./data/")
            np.save("./data/data_w_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).npy".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random), w)
            np.save("./data/data_p_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).npy".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random), p)
            np.save("./data/data_e_molecule_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).npy".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random), self.diagonal_H11)


### dynamics
    @staticmethod
    @njit(parallel=True)
    def psi_dot(hbar,psi,data,row_indice,col_indice):
        result = np.zeros(psi.shape[0], dtype=np.complex128)
        for i in range(len(row_indice)):
                row = int(row_indice[i])
                col = int(col_indice[i])
                result[row] += data[i]*psi[col]
        return 1/(1j*hbar) * result
    
    @staticmethod
    @njit
    def rk4(psi_dot,dt,hbar,psi, data,row_indice,col_indice):
        k1_psi = dt * psi_dot(hbar,psi, data,row_indice,col_indice)
        k2_psi = dt * psi_dot(hbar,psi+0.5*k1_psi, data,row_indice,col_indice)
        k3_psi = dt * psi_dot(hbar,psi+0.5*k2_psi, data,row_indice,col_indice)
        k4_psi = dt * psi_dot(hbar,psi+k3_psi, data,row_indice,col_indice)
        psi =  psi + (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi) / 6.
        return psi
    
    def get_psi_0(self):
        self.psi_0 = np.zeros(self.n_superblock, dtype=np.complex128)
        for q in range(-self.floquet_level, self.floquet_level+1):
            self.psi_0[(self.total_size*(q+self.floquet_level+1)-self.total_size):(self.total_size*(q+self.floquet_level+1)-self.total_size)+self.light_size] = 1

        self.psi_0 = self.psi_0.astype('complex128')/np.sqrt(self.light_size)

        return self.psi_0

    @staticmethod
    @njit
    def transform_list(original_list, m, t,omega,floquet_level):
        # 计算分组数量
        num_groups = original_list.shape[0] // m
        # 将原始列表转换为二维数组，每一行为一个分组
        grouped_array = np.asarray(original_list, dtype=np.complex128).reshape(num_groups, m)
        # 计算每个元素应该乘以的系数

        coefficients = np.asarray(np.exp(1j * omega * (np.arange(num_groups,dtype=np.float64) - floquet_level) * t), dtype=np.complex128)

        # 将系数扩展为二维数组，方便和分组数组相乘
        coefficients = coefficients.reshape(num_groups, 1)
        # 使用向量化操作进行乘法，并对每一列求和得到新列表
        new_list = np.asarray(np.sum(grouped_array * coefficients, axis=0) / num_groups, dtype=np.complex128)
        return new_list
    
    # @njit(parallel=True)
    def dynamics(self,data,row_indice,col_indice):
        TIME = np.zeros(self.steps, dtype=np.float64)
        psi_0_original  = self.transform_list(self.psi_0, self.total_size, 0,self.omega,self.floquet_level)
        # print("psi_0_original:",psi_0_original)
        observable = np.zeros(self.steps, dtype=np.complex128)
        psi_t = np.asarray(self.psi_0,dtype=np.complex128)
        for i in range(self.steps):
            TIME[i] = i*self.dt
            psi_t_original = self.transform_list(psi_t, self.total_size, TIME[i],self.omega,self.floquet_level)
            observable[i] = np.dot(np.conj(psi_0_original).T, psi_t_original)
            psi_t = self.rk4(self.psi_dot,self.dt,self.hbar,psi_t,data,row_indice,col_indice)

            
            # print(psi_t_original)
            
        return TIME, observable
    
    def run_dynamics(self):
        self.get_psi_0()
        row_indice,col_indice,data = self.sparse_matrix()
        TIME, observable = self.dynamics(data,row_indice,col_indice)
        return TIME, observable
    
### plot dynamics

    def plot_corr(self,TIME,observable,SAVE=False,PLOT=False):
        if PLOT:
            plt.figure(dpi=350)
            plt.plot(TIME,observable)
            plt.xlabel('time')
            plt.ylabel(r'$<\varphi(0)|\varphi(t)>$')
            plt.tight_layout()

            if not os.path.exists("./figs/"):
                os.makedirs("./figs/")
            plt.savefig('./figs/fig_CORR_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).png'.format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random))

        if SAVE:
            if not os.path.exists("./data/"):
                os.makedirs("./data/")
            np.save("./data/data_TIME_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).npy".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random), TIME)
            np.save("./data/data_CORR_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).npy".format(self.total_size, self.V_coupling_strength, self.omega, self.disorder, self.phase_up, self.phase_random), observable)

    def plot_both(self,TIME,observable,w,p,REAL=True,SAVE=False):
        plt.figure(dpi=350)
        DMD2 = np.vstack((TIME,observable)).T
        if REAL:
            plt.plot(
            2 * np.pi * np.fft.fftfreq(len(DMD2[:, 0]), (DMD2[1, 0] - DMD2[0, 0])),
            len(DMD2[:, 0]) * np.real(np.fft.ifft(DMD2[:, 1])) *
            (DMD2[1, 0] - DMD2[0, 0]), '-',label='FFT',color='blue')
        else:
            plt.plot(
            2 * np.pi * np.fft.fftfreq(len(DMD2[:, 0]), (DMD2[1, 0] - DMD2[0, 0])),
            len(DMD2[:, 0]) * np.imag(np.fft.isfft(DMD2[:, 1])) *
            (DMD2[1, 0] - DMD2[0, 0]), '-',label='FFT',color='blue')            

        plt.plot(w,p,c='red',label=r'$\{<\psi_i|(\delta(\tilde{\lambda}-\omega) \tilde{U}^{\dagger} P_0 \tilde{U})|\psi_i>\}$')
        # plt.plot(w_filtered2, P_filtered2,'--',c='green',label=r'$Tr(\delta(\lambda-w) P_0 )$')
        plt.xlim(self.left_lim,self.right_lim)

        plt.xlabel(r'$\omega$')
        plt.ylabel('Spectra')

        plt.legend()
        plt.tight_layout()

        if SAVE:
            if not os.path.exists("./figs/"):
                os.makedirs("./figs/")
            plt.savefig('./figs/fig_FULL_(totalsize={0})_(V={1})_(omega={2})_(disorder={3})_(phase_divide={4})_(phase_rand={5}).png'.format(self.total_size, self.V_coupling_strength, self.omega,self.disorder, self.phase_up, self.phase_random))