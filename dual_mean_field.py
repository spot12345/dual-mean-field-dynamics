import numpy as np
import torch
import math

def sk_model(n,seed):
    # n=100
    variance = 1 / n
    # seed = 20
    np.random.seed(seed)
    J = np.random.normal(loc=0, scale=math.sqrt(variance), size=(n, n))

    J = np.triu(J, k=1)
    J = J + J.T
    J=torch.tensor(J)
    return J


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
        #energy=0.5*torch.sum((torch.sign(self.x[:,0:self.N]) @self.J)*
        #                                    torch.sign(self.x[:,0:self.N]), 1)
        energy=0.5*torch.sum((self.x[:,0:self.N] @self.J)*
                                            self.x[:,0:self.N], 1)
        return energy


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
