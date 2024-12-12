
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import scipy as sp

colors = ['blue', 'darkorange','green','red','purple','brown','pink', 'gray', 'olive', 'cyan' , 'darkgreen']
epochs = 200
n_tasks = 1
L = 0
gamma0 = 2
widths = [4096]
device = 'mps'
gen = torch.Generator(device=device)
gen.manual_seed(1234)
batch = 25
eta = 0.1
Synt = True
relu = False
trials = 30
perc = 0.95

print(f'Synthetic data: {Synt}, perc: {perc}')

if Synt:
    filename = 'Losses/SYNT/'
else:
    filename = 'Losses/MNIST/'

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

avg_loss_th = torch.zeros((n_tasks+1, (n_tasks+1)*epochs))
avg_loss_exp = torch.zeros((n_tasks+1, (n_tasks+1)*epochs))

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

#     --------------------------  TRIALS CYCLES   -------------------------------------

for nt in range(trials):

    print(f'Trial number: {nt+1}')

    tasks = []
    tasks_test = []

    #lim = int((784-int(X.shape[1]*perc))/2)
    #for _ in range(n_tasks+1):
    #            par_perm = np.append(np.append(np.arange(0,lim,1),np.random.permutation(int(X.shape[1]*perc))+lim), np.arange(int(X.shape[1]*perc)+lim,784,1))
    #            #perm = np.random.permutation(X.shape[1])
    #            tasks.append( torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X.cpu(), perm=par_perm)).to(device) )
    #            tasks_test.append(torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X_test.cpu(), perm=par_perm)).to(device))

    lim = int(X.shape[1]*perc)
    for _ in range(n_tasks+1):
            par_perm =np.append(np.random.permutation(int(X.shape[1]*perc)),np.arange(lim,X.shape[1],1))
            #perm = np.random.permutation(X.shape[1])
            tasks.append( torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X.cpu(), perm=par_perm)).to(device) )
            tasks_test.append(torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X_test.cpu(), perm=par_perm)).to(device))

    def plot_dist(vecs, all= True, log= True):
        fig = plt.figure(figsize=(16,10))
        for i,vec in enumerate(vecs):
            counts, bins = torch.histogram(vec.view(-1), bins=200)
            if i == 0: 
                plt.plot(bins[:-1], counts/(torch.sum(counts)*(bins[1]-bins[0])), linestyle = '-.', color='black', label = 'Theory')
            else:
                plt.plot(bins[:-1], counts/(torch.sum(counts)*(bins[1]-bins[0])), color='purple', alpha=0.4, label = 'Experimental', linewidth=3)


            if not all:
                if log:
                    plt.yscale('log')
                plt.grid()
                plt.xlabel('$x$')
                plt.ylabel('$p(x)$')
                plt.show()
        if all:
            if log:
                plt.yscale('log')
            plt.xlabel('$z$', fontsize=16)
            plt.ylabel('$p(z)$',fontsize=18)
            plt.xticks(size=14)
            plt.yticks(size=14)
            plt.grid()
            plt.legend(prop={'size':16})
            plt.show()

    def compare(phi, phi2):
        fig, ax = plt.subplots(1,2)
        a1 = ax[0].imshow(phi, cmap='coolwarm')
        a2 = ax[1].imshow(phi2, cmap='coolwarm')
        plt.colorbar(a1)
        plt.colorbar(a2)
        plt.show()

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

    def updateDelta(ht, hc, z, K, delta, deltac, eta):

        Phi = get_Phi(ht, hc, relu=relu)
        G = get_G(z, ht, hc, relu=relu)
        ntk = Phi + G * K
        del_new = delta - eta/P * ntk @ deltac

        return del_new

    def compare_NTK(idxs):
        for idx in idxs:
            Phi_exp = h2[idx] @ h[idx].T/N
            G_exp = g2[idx] @ g[idx].T/N
            Phi_th = Phis2[idx]
            G_th = Gs2[idx]
            ntk_exp = (Phi_exp + G_exp*K_x.to(device)).cpu()
            ntk_th = (Phi_th.to(device) + G_th.to(device)*K_x.to(device)).cpu()
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].set_title('Expr')
            ax[1].set_title('Theory')
            exp = ax[0].imshow(ntk_exp.cpu(), cmap='coolwarm')
            th = ax[1].imshow(ntk_th.cpu(), cmap='coolwarm')
            plt.colorbar(exp)
            plt.colorbar(th)
            plt.show()

    def plotKernel(idx,idx2, time, relu=relu):

            time_idx = int(time/epochs)
            row_titles = ['$\Phi$', '$G$', '$K^{NTK}$']

            Phi0_exp = get_Phi(h[idx, 0],h[idx2, 0], relu)
            G0_exp = get_G(z[0, 0],h[idx, 0],h[idx2, 0], relu)
            Phi_exp = get_Phi(h[idx,time],h[idx2,time], relu)
            G_exp = get_G(z[0, time], h[idx, time], h[idx2, time], relu)

            Phi0 = get_Phi(hs[idx, 0],hs[idx2, 0], relu)
            G0 = get_G(zs[0, 0],hs[idx, 0],hs[idx2, 0], relu)
            PhiT = get_Phi(hs[idx, time],hs[idx2, time], relu)
            GT = get_G(zs[0, time],hs[idx, time],hs[idx2, time], relu)

            fig, ax = plt.subplots(3,4, figsize=(15,8))

            ax1 = ax[0,0].imshow(Phi0, cmap='coolwarm')
            ax[0,0].set_title('Theory $t=0$',fontsize=16, pad=20)

            ax2 = ax[0,1].imshow(Phi0_exp, cmap='coolwarm')
            ax[0,1].set_title('Exp. $t=0$',fontsize=16, pad=20)

            ax3 = ax[0,2].imshow(PhiT, cmap='coolwarm')
            ax[0,2].set_title(f'Theory $t=t_{{{time_idx}}}$',fontsize=16, pad=20)

            ax4 = ax[0,3].imshow(Phi_exp, cmap='coolwarm')
            ax[0,3].set_title(f'Exp. $t=t_{{{time_idx}}}$',fontsize=16, pad=20)

            ax5 = ax[1,0].imshow(G0, cmap='coolwarm')
            #ax[1,0].set_title('$G_{Th.}(0)$')

            ax6 = ax[1,1].imshow(G0_exp, cmap='coolwarm')
            #ax[1,1].set_title('$G_{Exp.}(0)$')

            ax7 = ax[1,2].imshow(GT, cmap='coolwarm')
            #ax[1,2].set_title(f'$G_{{Th.}}(t_{{{time_idx}}})$')

            ax8 = ax[1,3].imshow(G_exp, cmap='coolwarm')
            #ax[1,3].set_title(f'$G_{{Exp.}}(t_{{{time_idx}}})$')

            ax9 = ax[2,0].imshow(Phi0.cpu() + G0.cpu() * Ks[idx,idx2].cpu(), cmap='coolwarm')
            #ax[2,0].set_title('$K^{NTK}_{Th.}(0)$')

            ax10 = ax[2,1].imshow(Phi0_exp.cpu() + G0_exp.cpu() * Ks[idx,idx2].cpu(), cmap='coolwarm')
            #ax[2,1].set_title('$K^{NTK}_{Exp.}(0)$')

            ax11 = ax[2,2].imshow(PhiT.cpu() + GT.cpu() * Ks[idx,idx2].cpu(), cmap='coolwarm')
            #ax[2,2].set_title(f'$K^{{NTK}}_{{Th.}}(t_{{{time_idx}}})$', fontsize=18)

            ax12 = ax[2,3].imshow(Phi_exp.cpu() + G_exp.cpu() * Ks[idx,idx2].cpu(), cmap='coolwarm')
            #ax[2,3].set_title(f'$K^{{NTK}}_{{Exp.}}(t_{{{time_idx}}})$')

            axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]
            for i,a in enumerate(axes):
                ax[int(i/4), i%4].axis('off')
                plt.colorbar(a, ax=ax[int(i/4), i%4])

            for j, row_title in enumerate(row_titles):
                fig.text(0.02, (3-j)/3.0 - 2/11, row_title, va='center', ha='center', fontsize=25, rotation='vertical')

            plt.tight_layout()
            plt.show()

    def computeCorr(h1,h2,z,typekern):
        if typekern == 'h':
            kernels = torch.einsum('ijkl, ijkd -> ijld', h1*(h1>0)/S, h2*(h2>0))
        elif typekern == 'g':
            kernels = torch.einsum('ijkl, ijkd -> ijld', torch.stack([z] * len(X), dim=3)*(h1>0)/S, torch.stack([z] * len(X), dim=3)*(h2>0))
        norms = torch.norm(kernels, dim=(2, 3))
        norm = torch.einsum('ij,ik -> ijk', norms, norms)
        corr = torch.einsum('ijkl, iakl -> ija', kernels, kernels) / norm

        fig, ax = plt.subplots(2,3,figsize=(16,8))
        vmin = torch.min(corr)
        vmax = torch.max(corr)

        for i in range(2):
            for j in range(3):
                a = ax[i,j].imshow(corr[3*i+j], cmap='coolwarm', vmin=vmin, vmax=vmax)
                ax[i,j].invert_yaxis()
                if j==2:
                    cbar = plt.colorbar(a, ax=ax[i,j])
                    cbar.ax.tick_params(labelsize=22) 
                ax[i,j].set_title(f'$\mathcal{{T}}_{{{(i*3)+j+1}}}$',fontsize=26)
                for d in range(n_tasks+1):
                    ax[i,j].axvline(d*epochs, color='black', alpha=0.85, linestyle=':')
                    ax[i,j].axhline(d*epochs, color='black', alpha=0.85, linestyle=':')
                    ax[i,j].xaxis.set_tick_params(labelbottom=False)
                    ax[i,j].yaxis.set_tick_params(labelbottom=False)
                    if i==1:
                        ax[i,j].set_xlabel('Epochs',fontsize=26)
                        ax[i,j].xaxis.set_tick_params(labelbottom=True)
                    if j==0:
                        ax[i,j].set_ylabel('Epochs',fontsize=26)
                        ax[i,j].yaxis.set_tick_params(labelbottom=True)
                    ax[i,j].tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()
        plt.show()

    def DMFTalignment(h1th,h2th, h1exp,h2exp, zth, zexp, typekern='h'):
        if typekern == 'h':
            kernelsth = torch.einsum('ijkl, ijkd -> ijld', h1th*(h1th>0)/S, h2th*(h2th>0))
            kernelsexp = torch.einsum('ijkl, ijkd -> ijld', h1exp*(h1exp>0)/S, h2exp*(h2exp>0))

        elif typekern == 'g':
            kernelsth = torch.einsum('ijkl, ijkd -> ijld', torch.stack([torch.stack([zth] * len(X), dim=3).squeeze()] * (n_tasks+1), dim=0)*(h1th>0)/S, torch.stack([torch.stack([zth] * len(X), dim=3).squeeze()] * (n_tasks+1), dim=0)*(h2th>0))
            kernelsexp = torch.einsum('ijkl, ijkd -> ijld', torch.stack([torch.stack([zexp] * len(X), dim=3).squeeze()] * (n_tasks+1), dim=0)*(h1exp>0)/S, torch.stack([torch.stack([zexp] * len(X), dim=3).squeeze()] * (n_tasks+1), dim=0)*(h2exp>0))

        A = torch.einsum('ijkd, ijkd -> ij' , kernelsth , kernelsexp)
        normth = torch.sqrt(torch.einsum('ijkd, ijkd -> ij' , kernelsth , kernelsth))
        normexp =  torch.sqrt(torch.einsum('ijkd, ijkd -> ij' , kernelsexp , kernelsexp))

        normtot = normth * normexp

        align = A / normtot

        return align

    def ForgMeasure(hs1, hs2, zs, t1, t2):
        h1 = hs1[t1]
        h2 = hs2[t2]
        z = zs[0]
        S = hs1.shape[2]

        H = torch.einsum('jkl, jkd -> jld', h1*(h1>0)/S, h2*(h2>0))
        G = torch.einsum('jkl, jkd -> jld', torch.stack([z] * len(X), dim=2)*(h1>0)/S, torch.stack([z] * len(X), dim=2)*(h2>0))
        K = H + G * Ks[t1,t2]

        KKT = torch.einsum('ijk, ikj -> ijk', K, K)
        forg = torch.norm(KKT - torch.eye(P), dim=([1,2]))
        return forg

    save_out = False
    D = X.shape[1]
    P = len(X)

    for regime in ['mup']:
        for N in widths:

            h = torch.empty((n_tasks+1,(n_tasks+1)*epochs, N, P))
            z = torch.empty((1,(n_tasks+1)*epochs, N))
            loss = torch.empty((n_tasks+1, (n_tasks+1)*epochs))

            acc = []    

            mlp = MLP(N,L,regime, gamma0)
            lrs = {'sp':1,'ntk':1,'mup': eta*mlp.gamma**2}
            lr = lrs.get(regime)

            if regime == 'ntk' or regime == 'mup':
                mlp = mlp.apply(init_weights)

            #summary(mlp, (1,784))
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

            avg_loss_exp += loss.clone().detach()/trials

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

    S = 10000

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

    avg_loss_th += torch.mean(deltas**2, axis= 2)*0.5/trials          

#plotKernel(0,1,2*epochs-1)
torch.save(avg_loss_exp, filename+f'losses_{perc}_exp')
torch.save(avg_loss_th ,filename+f'losses_{perc}_th')

print('Losses saved, work done!')