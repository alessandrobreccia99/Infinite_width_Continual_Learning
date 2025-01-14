
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


gammas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4]
percents = [0.5]
widths = [8,16,32,64,128,256,512,1024,2048,4096]


colors = ['blue', 'darkorange','green','red','purple','brown','pink', 'gray', 'olive', 'cyan' , 'darkgreen']
epochs = 200
n_tasks = 1
L = 0
device = 'mps'
gen = torch.Generator(device=device)
gen.manual_seed(123)
batch = 25
eta = 0.1
Synt = True
relu = True
trials = 10
regime = 'mup'
S = 10000

if Synt:
    if relu:
        filename = 'Losses/SYNT_RELU/'
    else:
        filename = 'Losses/SYNT/'
else:
    if relu:
        filename = 'Losses/MNIST_RELU/'
    else:
        filename = 'Losses/MNIST/'

avg_loss_th = torch.zeros((len(gammas), len(widths) , n_tasks+1, (n_tasks+1)*epochs))
avg_loss_exp = torch.zeros((len(gammas), len(widths) , n_tasks+1, (n_tasks+1)*epochs))

print(f'Synthetic data: {Synt}')

class MLP(nn.Module):
                    def __init__(self, w, L, param, gam):
                        super(MLP, self).__init__()
                        self.w = w
                        if param =='ntk':
                            self.gamma = gam
                            self.in_scale = 784**0.5
                            self.out_scale = self.w**0.5*self.gamma
                        elif param == 'mup': 
                             self.gamma = gam*self.w**0.5
                             self.in_scale = 784**0.5
                             self.out_scale = self.w**0.5*self.gamma
                        elif param == 'sp':
                             self.gamma = 1
                             self.in_scale = 1
                             self.out_scale = 1

                        self.fc1 = nn.Linear(784, self.w, bias=False)
                        self.fc2 = nn.Linear(self.w, 1, bias=False)
                        self.relu = nn.ReLU()
                        self.L = L

                    def forward(self, x):
                        h1 = self.fc1(x)/self.in_scale
                        if relu:
                            h1act = self.relu(h1)
                        else:
                            h1act = h1
                        z = ( torch.sum(self.fc2.weight.T, dim=1)).detach().clone()
                        h2 = self.fc2(h1act)/self.out_scale

                        return h2, h1.detach().clone(), z

@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.normal_()

def permut_row(x, perm):
            return x[perm]

       # ---------------------- START DATA -------------------------
if not Synt:
        tasks = []
        tasks_test = []
        for dig1, dig2 in zip([0],[1]):
            data = pd.read_csv('~/data/MNIST/mnist_train.csv')
            test = pd.read_csv('~/data/MNIST/mnist_test.csv')
            data = data[data['label'].isin([dig1, dig2])]
            test = test[test['label'].isin([dig1, dig2])]
            X = torch.tensor(data.drop('label', axis = 1).to_numpy(), device=device)/255
            X_test = torch.tensor(test.drop('label', axis = 1).to_numpy(), device=device)/255
            X = X[:batch]
            Y_temp = torch.tensor(data['label'].to_numpy(), device=device)
            Y = torch.tensor([[y*1.0-torch.min(Y_temp)] for y in Y_temp], device=device)
            Y = Y[:batch]
            _ , indeces = torch.sort(Y_temp[:batch])
            Y = Y[indeces]
            X = X[indeces]
            Y_temp = torch.tensor(test['label'].to_numpy(), device=device)
            Y_test = torch.tensor([[y*1.0 - torch.min(Y_temp)] for y in Y_temp], device=device)
            #tasks.append(X)
            #tasks_test.append(X_test)
else:
        tasks = []
        tasks_test = []
        P = 25
        D = 784
        X = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D)).sample((P,)).to(device)
        X_test = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D)).sample((P,)).to(device)
        Y = torch.hstack((torch.zeros((int(P/2)+1,)),torch.ones((int(P/2),)))).view(25,1).to(device)
        Y_test = Y.clone()

tasks = []
tasks_test = []
lim = int(X.shape[1]*percents[0])
for _ in range(n_tasks+1):
        par_perm =np.append(np.random.permutation(int(X.shape[1]*percents[0])),np.arange(lim,X.shape[1],1))
        #perm = np.random.permutation(X.shape[1])
        tasks.append( torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X.cpu(), perm=par_perm)).to(device) )
        tasks_test.append(torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X_test.cpu(), perm=par_perm)).to(device))

def sampleChi(K, P, S):
        return torch.distributions.MultivariateNormal(torch.zeros(P), K).sample((S,))
# -----------------------------------------------------------------------------------------
def sampleXi(S):
    return torch.distributions.Normal(0,1).sample((S,))
# -----------------------------------------------------------------------------------------
def get_Phi(h1,h2,relu=relu):
    if relu:
        h1 = h1 * (h1>0)
        h2 = h2 * (h2>0)
    N = h1.shape[0]
    return h1.T @ h2 / N
# -----------------------------------------------------------------------------------------
def get_G(z1, h1, h2, relu=relu):
    N = z1.shape[0]
    if relu:
        g1 = torch.stack([z1] * len(X), dim=1) * (h1>0) 
        g2 = torch.stack([z1] * len(X), dim=1) * (h2>0)
    else:
        return z1.T @ z1 / N
    return g1.T @ g2 / N
# -----------------------------------------------------------------------------------------
def Update( delta, zup, hup, z, h, K, gamma0, eta, relu=relu):
    P = delta.shape[0]
    if relu:
        g = torch.stack([z] * len(X), dim=1) * (h > 0)
        hact1 = h * (h > 0)
    else:
        g = torch.stack([z] * len(X), dim=1)
        hact1 = h
    dh = torch.einsum('ij, kj -> ik' ,torch.einsum('ij, j -> ij', g , delta), K )
    dz = torch.einsum('ij, j -> i', hact1 , delta)
    #dh = g @ (K.T * torch.stack([delta] * len(X), dim=1))
    #dz = hact1 @ delta
    h_up = hup + eta*gamma0/P*dh
    z_up = zup + eta*gamma0/P*dz
    return h_up, z_up
# -----------------------------------------------------------------------------------------
def updateDelta(ht, hc, z, K, delta, deltac, eta):

    Phi = get_Phi(ht, hc, relu=relu)
    G = get_G(z, ht, hc, relu=relu)
    ntk = Phi + G * K
    del_new = delta - eta/P * ntk @ deltac
    return del_new

for g0,gamma0 in enumerate(gammas):
     for per,perc in enumerate(percents):
        for wi,N in enumerate(widths):
            print(f'Gamma0 = {gamma0}, width: {N}')

            #     --------------------------  TRIALS CYCLES   -------------------------------------

            for nt in range(trials):
            
                print(f'Trial number: {nt+1}')

                save_out = False
                D = X.shape[1]
                P = len(X)
                    
                h = torch.empty((n_tasks+1,(n_tasks+1)*epochs, N, P))
                z = torch.empty((1,(n_tasks+1)*epochs, N))
                loss = torch.empty((n_tasks+1, (n_tasks+1)*epochs))
                acc = []    
                mlp = MLP(N,L,regime, gamma0)
                lrs = {'sp':1,'ntk':1,'mup': eta*mlp.gamma**2}
                lr = lrs.get(regime)
                if regime == 'ntk' or regime == 'mup':
                    mlp = mlp.apply(init_weights)

                mlp = mlp.to(device)
                optimizer = torch.optim.SGD(mlp.parameters(), lr= lr )
                eos = 2/lr
                MSE = nn.MSELoss()
                for t,Xt in enumerate(tasks):        
                        for epoch in range(epochs):
                                    if epoch==0 and t==0:
                                        out, _ , _ = mlp(Xt)
                                        out_at_0 = out.detach().clone()
                                    running_loss = 0.0
                                    for n,X in enumerate(tasks):
                                        out, ht, zt = mlp(X)
                                        h[n,t*epochs+epoch] = ht.T
                                        z[0,t*epochs+epoch] = zt.T
                                        loss[n,t*epochs+epoch] = MSE(out,Y)*0.5
                                    optimizer.zero_grad()
                                    out, _ , _ = mlp(Xt)
                                    losscurr = MSE(out-out_at_0, Y)*0.5
                                    losscurr.backward()
                                    running_loss += losscurr.item()
                                    optimizer.step()
                        #print(f'Finished Training task{t}, train loss: {running_loss/batch}')
                        acct = []
                        for s in range(t+1):
                            acct.append( (torch.sum(torch.round(mlp(tasks_test[s])[0]) == Y_test)/len(Y_test)).item() )  
                        acc.append(acct) 

                avg_loss_exp[g0,wi] += loss.clone().detach()/trials

                if save_out:   
                    torch.save(loss, f'loss_data/loss{N}_gamma{gamma0}.pt')

                mlp = mlp.eval()
                #print((torch.sum(torch.round(mlp(X_test)[0]) == Y_test)/len(Y_test)).item())
                #print((torch.sum(torch.round(mlp(tasks_test[1])[0]) == Y_test)/len(Y_test)).item())

                D = X.shape[1]
                N = widths[-1]
                P = len(X)

                X_tot = tasks[0]
                for i in range(n_tasks):
                    X_tot = torch.vstack((X_tot, tasks[i+1]))

                K_tot = X_tot @ X_tot.T / D

                Ks = torch.empty((n_tasks+1, n_tasks+1, P, P))
                for i in range(n_tasks+1):
                    for j in range(n_tasks+1):
                       Ks[i,j] = tasks[i] @ tasks[j].T / D

                #plt.imshow(K_tot.cpu(), cmap='coolwarm')
                #plt.colorbar()
                #plt.show()

                hs = torch.empty((n_tasks+1,(n_tasks+1)*epochs, S, P))
                zs = torch.empty((1,(n_tasks+1)*epochs, S))
                deltas = torch.empty((n_tasks+1,(n_tasks+1)*epochs, P))

                zs[0,0] = sampleXi(S)
                chi_tot = sampleChi( K_tot.cpu(), (n_tasks+1)*P, S)

                for t in range(n_tasks+1):
                    hs[t,0] = chi_tot[::,P*t:P*(t+1)]
                    deltas[t,0] = torch.sum(Y, dim=1).cpu()

                for c in range(n_tasks+1):
                    for i in np.arange(c*epochs-1,(c+1)*epochs-1,1):
                        for t in range(n_tasks+1):
                            if i == -1: continue

                            h_up, z_up = Update( deltas[c,i] , zs[0,i], hs[t,i], zs[0,i], hs[c,i], Ks[t,c], gamma0, eta, relu=relu)
                            hs[t,i+1] = h_up
                            zs[0,i+1] = z_up

                            delta_up = updateDelta( hs[t,i], hs[c,i], zs[0,i], Ks[t,c], deltas[t,i], deltas[c,i], eta )
                            deltas[t,i+1] = delta_up

                avg_loss_th[g0,wi] += torch.mean(deltas**2, axis= 2)*0.5/trials          

            print(f'Losses saved, work gamma0 = {gamma0} and perc = {perc} done!')


torch.save(avg_loss_exp, filename+f'losses_gamma-widths_2_exp')
torch.save(avg_loss_th ,filename+f'losses_gamma-widths_2_th')