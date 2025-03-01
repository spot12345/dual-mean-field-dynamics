import numpy as np
import torch
import math
import random
import networkx as nx
from matplotlib import pyplot as plt
import re

def load_file(test_instance):
    G = nx.Graph()
    # load Gset instance file
    path = './Gset/' + test_instance  # file path
    with open(path, "r") as f:
        l = f.readline()
        N, m = [int(x) for x in l.split(" ") if x != "\n"]  # n: nodes, m:edges
        W = np.zeros([N, N])
        G.add_nodes_from(range(N))
        for k in range(m):
            l = f.readline()
            l = l.replace("\n", "")
            i, j, w = [int(x) for x in l.split(" ")]
            W[i - 1, j - 1], W[j - 1, i - 1] = w, w
            G.add_edge(i - 1, j - 1)
    J = -W.copy()
    with open('./targetvalue.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    target_value = re.findall(".*" + test_instance + " (.*).*", content)[0]
    return G, J, float(target_value)
class transverse:
    def __init__(
            self, J, beta, trials, gama, g,a_set,Delta_t=0.1, dev='cuda:0', c0=0.5, dtype=torch.float64, seed=1,FIG_energy='True'
    ):

        self.dtype = dtype
        self.dev = dev
        self.seed = seed
        self.J = J.to(self.dtype).to(self.dev)
        self.w2 = (self.J * self.J).sum()
        self.beta = torch.from_numpy(beta).to(self.dtype).to(self.dev)
        self.N = self.J.shape[0]
        self.trials = trials
        self.m = J.sum() / 2
        self.N_step = self.beta.shape[0]
        self.J_var = (sum(sum(J * J)) / (self.N * (self.N - 1))) ** 0.5
        self.sum_w = -J.sum() / 4
        self.gama = gama
        self.Delta_t = Delta_t
        self.FIG_energy=FIG_energy
        self.a_set=a_set
        self.g=g
        self.c0=c0



    def calculate_energy(self):
        cut = -self.sum_w - 0.25 * torch.sum((torch.sign(self.x[:,0:self.N]) @self.J)*
                                            torch.sign(self.x[:,0:self.N]), 1)
        return cut


    def iterate_T(self):
        torch.manual_seed(self.seed)
        self.x = torch.randn([self.trials, self.N], device=self.dev, dtype=self.dtype) * 0.01
        self.y = torch.randn([self.trials, self.N], device=self.dev, dtype=self.dtype) * 0.01
        self.a0 = 1
        self.a = torch.linspace(-2, self.a_set, self.N_step, dtype=self.dtype)
        #
        ## self.a = 0.2 * (torch.exp(torch.linspace(0, 2.7, self.N_step)) - 1) - 2
        ## self.a = 0.5 * (torch.log(torch.linspace(1, 4, self.N_step))) - 2


        self.a[self.a > self.a0] = self.a0
        self.a = self.a.to(self.dev)
        # self.a = 0.5*torch.ones(self.N_step)
        if self.FIG_energy == 'True':

            energy = torch.zeros([self.trials, self.N_step])
            Track=torch.zeros([self.N_step+1,self.N])
            for jj in range(self.N_step):
                # self.z = torch.matmul(torch.sign(self.x), self.J)
                self.z = torch.matmul(self.x, self.J)
                self.cache = self.Delta_t * (self.x * self.x * self.x - self.a[jj] * self.x + self.c0 * self.z)
                self.x = self.x - self.g * self.cache + self.gama * self.Delta_t * self.y
                self.y = self.y - self.g*self.Delta_t * self.y - self.gama * self.cache

                self.x = torch.where(self.x > 1, 1, self.x)
                self.x = torch.where(self.x < -1, -1, self.x)

                # print(self.x)
                if self.trials==1:

                    Track[jj+1,:]=self.x

                energy[:, jj] = self.calculate_energy()

            self.x.requires_grad_(False)
            return energy,Track

        elif self.FIG_energy == 'False':
            # energy=0
            for jj in range(self.N_step):
                # self.z = torch.matmul(torch.sign(self.x), self.J)
                self.z = torch.matmul(self.x, self.J)
                self.cache = self.Delta_t * (self.x * self.x * self.x - self.a[jj] * self.x + 0.5 * self.z)
                # self.ci=self.J.sum(1)

                self.x = self.x - self.g * self.cache + self.gama * self.Delta_t * self.y
                self.y = self.y - self.g*self.Delta_t * self.y - self.gama * self.cache

                self.x = torch.where(self.x > 1, 1, self.x)
                self.x = torch.where(self.x < -1, -1, self.x)

            self.x.requires_grad_(False)
            return self.calculate_energy()



if __name__ == "__main__":

    G, J, target = load_file('G55')
    J = torch.tensor(-J)

    n_step = 5000

    trials = 1000
    beta = np.exp(np.linspace(math.log(0.1), math.log(20), n_step))

    tran = transverse(J, beta, trials, a_set=2, gama=0, Delta_t=0.23,g=1, dev='cuda:0', c0=0.5,
                                  dtype=torch.float64,
                                  seed=1, FIG_energy='False')
    energy = tran.iterate_T()
    mean=torch.sum(energy, dim=0) / trials
    max=torch.max(energy)

##########meaning: hyper-parameter=[name,g=[g of pure gradient,g of pure rotational, g of OGM],gama=.......,
#                                   Delta_t=[Delta_t of pure gradient,Delta_t of pure rotational, Delta_t of OGM],n_step=......]
#
#
#####hyper-parameter=[[name=G55,g=[1,0,0.51],gama=[0,1,1],Delta_t=[0.23,0.504,0.55],n_step=[5000,10000,10000]
#                     [name=G56,g=[1,0,0.1],gama=[0,1,1],Delta_t=[0.429,0.0531,0.184],n_step=[20000,20000,20000]
#                     [name=G57,g=[1,0,0.55],gama=[0,1,1],Delta_t=[0.12,1.27,0.96],n_step=[20000,20000,20000]]
#                     [name=G58,g=[1,0,0.1],gama=[0,1,1],Delta_t=[0.1,0.09,0.22],n_step=[20000,20000,20000]]
#                     [name=G59,g=[1,0,1],gama=[0,1,0.5],Delta_t=[0.41,0.9,0.48],n_step=[50000,20000,100000]]
#                     [name=G60,g=[1,0,0.1],gama=[0,1,1],Delta_t=[0.21,0.03,0.2],n_step=[20000,20000,100000]]
#                     [name=G61,g=[1,0,0.01],gama=[0,1,1],Delta_t=[0.1,0.05,0.04],n_step=[20000,100000,100000]]
#                     [name=G62,g=[1,0,0.01],gama=[0,1,1],Delta_t=[0.28,0.27,0.27],n_step=[20000,100000,100000]]
#                     [name=G63,g=[1,0,0.02],gama=[0,1,1],Delta_t=[00.07,0.1,0.07],n_step=[20000,20000,20000]]
#                     [name=G64,g=[1,0,1],gama=[0,1,0.5],Delta_t=[0.39,2.2,1.31],n_step=[100000,100000,100000]]
#                     [name=G65,g=[1,0,0.1],gama=[0,1,1],Delta_t=[0.53,0.3,0.38],n_step=[100000,100000,100000]]
#                     [name=G66,g=[1,0,0.1],gama=[0,1,1],Delta_t=[0.3,0.31,0.3],n_step=[20000,100000,100000]]
#                     [name=G67,g=[1,0,0.1],gama=[0,1,1],Delta_t=[0.5,0.31,0.33],n_step=[20000,100000,100000]]

# ]