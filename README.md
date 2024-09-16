# Master Thesis in Physics of Data, University of Padua & ETH Zurich 
![image](https://github.com/user-attachments/assets/8cea1cf4-f75a-47c9-be73-a7087f6a4a50 ){width=250}
![image](https://github.com/user-attachments/assets/791131d6-e0a1-4c3e-b044-6b56c6df4ade ){width=250}
<img src="[https://github.com/favicon.ico](https://github.com/user-attachments/assets/8cea1cf4-f75a-47c9-be73-a7087f6a4a50)" width="48">

## Continual Learning 
Continual Learning is a specific framework of **machine learning** in which *multiple tasks* are fed to the network in different training processes.
The objective is to obtain good perfomances on all tasks. The main issue is the so called **catastrophic forgetting**, the phenomenon due to which old tasks performances 
decreses as new tasks are trained. 

## Infinite Width Limit 
To characterize the network training dynamics, in order to address the cause of forgetting, we inspected a specific setting that allows us to pass from a stochastic high-dimensional 
Markovian process to a lower-dimensional, non-Markovian process. This is the **infinite width limit** in which a network, parametrized by vector $\boldsymbol{\theta}=\{W^0,...,W^L\}$
with depth L, has the number of hidden neurons per layer equal to N, that is taken to infinity.

## NTP
The Neural Tangent Parametrization (NTP) does not allow the NTK to grow and dynamically adapt to data, thus we do not have feature learning.

## MUP
In the Maximum Update Parametrization ( $\mu$ P) we scale the output with factor $\frac{1}{\gamma}$ and set the learning rate $\eta$ to $\gamma^2$. Using a dynamical mean field
theory approach we can obtain the update equation ruling the evolution of fields belonging to each task at any time in any training. 

# Results

### Distributions for 6 tasks
![08163c56-149d-41e8-ac95-cb772d098873](https://github.com/user-attachments/assets/cbefc0ed-a10a-4dc7-be2e-638ddfc48247)

### Evolution of $\Phi_{12}$ kernel
![2cbda33a-0c61-4cb1-9b83-f74a0cb294e4](https://github.com/user-attachments/assets/bfb2aa96-9321-4f54-9908-9020958dc8a0)

### Evolution of $G_{12}$ kernel
![53d54012-da0d-4d83-b6e5-68212cb16e45](https://github.com/user-attachments/assets/c1e8a377-401a-4578-918d-166e070c502c)

### Evolution of $K^{NTK}_{12}$ kernel
![afb84085-7ffc-4799-8e6d-a0250ab795c3](https://github.com/user-attachments/assets/0e3cb066-3e5e-4487-88c0-a9d82d84df8f)

### Losses evolution
![278c7b56-4f07-49b8-9a07-abaae1ded62b](https://github.com/user-attachments/assets/4ca9b792-5b6c-40e9-a2ef-6e8a876e972a)


