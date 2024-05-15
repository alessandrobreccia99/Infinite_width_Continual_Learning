import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyhessian import hessian
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

class MLP(nn.Module):
    """Standard MLP"""
    def __init__(self, w, L):
        super(MLP, self).__init__()
        self.w = w
        self.fc1 = nn.Linear(784, self.w, bias=False)
        self.layers = nn.ModuleList(nn.Linear(self.w,self.w, bias=False) for _ in range(L))
        self.fc2 = nn.Linear(self.w, 1, bias=False)
        self.relu = nn.ReLU()
        self.L = L

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for l in self.layers:
            x = self.relu(l(x))
        x = self.fc2(x)
        #x = x/np.sqrt(self.w*784)

        return x
    
def permut_row(x, perm):
    return x[perm]

epochs = 50
n_tasks = 3
N = 128
eos = 4 
L = 3
loss_hist = []
lam = []
acc = []
all = []
device = 'mps'
gen = torch.Generator(device=device)
gen.manual_seed(123)

# ---------------------- START DATA -------------------------
data = pd.read_csv('~/data/MNIST/mnist_train.csv')
test = pd.read_csv('~/data/MNIST/mnist_test.csv')
#data = data[data['label'].isin([0, 1])]
#test = test[test['label'].isin([0, 1])]
X = torch.tensor(data.drop('label', axis = 1).to_numpy(), device=device)/255
X_test = torch.tensor(test.drop('label', axis = 1).to_numpy(), device=device)/255

Y_temp = torch.tensor(data['label'].to_numpy(), device=device)
Y = torch.eye(1, device=device)[Y_temp]

Y_temp = torch.tensor(test['label'].to_numpy(), device=device)
Y_test = torch.eye(1, device=device)[Y_temp]

tasks = [X]
tasks_test = [X_test]

for _ in range(n_tasks):
        perm = np.random.permutation(X.shape[1])
        tasks.append( torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X.cpu(), perm=perm)).to(device) )
        tasks_test.append(torch.tensor(np.apply_along_axis(permut_row, axis = 1, arr=X_test.cpu(), perm=perm)).to(device))

# ----------------------- END DATA -----------------------------

mlp = MLP(N, L)
summary(mlp, (1,784))
mlp = mlp.to(device)

optimizer = torch.optim.SGD(mlp.parameters(), lr=2/eos)

def top_eigen(model, loss, X, Y, prt=False):

    hess_comp = hessian(model, loss, (X,Y) )
    top_eigenvalues, top_eigenvector = hess_comp.eigenvalues()
        
    return top_eigenvalues[-1] , top_eigenvector

MSE = nn.MSELoss()

s, _ = top_eigen(mlp, MSE , X, Y)

def GSalign(model, eigen):
   z = 0
   for g,p in zip(model.parameters(), eigen[0]):
        if g.requires_grad:
          gd = g.grad
          z += torch.sum(gd*p)/(torch.sqrt(torch.sum(gd*gd))*torch.sqrt(torch.sum(p*p))*len(eigen[0]))

   return z.item()

batch = len(X)

for t,Xt in enumerate(tasks):        
        for epoch in range(epochs):

                running_loss = 0.0
                for i in range(len(Xt)//batch):

                    # Batch of training 
                    ix = torch.randint(0, len(X), (batch,), generator=gen, device=device)

                    ixc = torch.randint(0, len(X), (1024,), generator=gen, device=device)

                    lt = []
                    for s in range(t):
                        sharp, eigen = top_eigen(mlp, MSE, tasks[s][ixc], Y[ixc])
                        lt.append(sharp)
                    lam.append(lt)    

                    optimizer.zero_grad()

                    out = mlp(Xt[ix])
                    loss = MSE(out, Y[ix])
                    loss.backward()
                    running_loss += loss.item()

                    #all.append(GSalign(mlp,eigen))

                    optimizer.step()
                    loss_hist.append(loss.item())
                    print(f'task {t} : (epoch: {epoch}), sample: {batch*(i+1)}, ---> train loss = {loss.item():.4f}')

        print(f'Finished Training task{t}')
        acc.append( torch.sum(torch.argmax(mlp(X_test), dim=1) == torch.argmax(Y_test, dim=1))/len(Y_test) )    

with open('data.txt', 'w') as file:  
     file.write(lam)
                        