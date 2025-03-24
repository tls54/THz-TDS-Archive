import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .back_prop_utils import c

## TODO: 
# - Add optimal param dict
# - Add Scheduler  

class FrequencyDependentModel(torch.nn.Module):
    def __init__(self, w_tensor, d, ICs_n, ICs_k):
        super().__init__()
        self.w_tensor = w_tensor  # Frequency tensor
        self.d = d  # Thickness of the material
        
        # Initialize n and k as trainable parameters (one per frequency point)
        self.n = torch.nn.Parameter(torch.tensor(ICs_n, dtype=torch.float32).repeat(len(w_tensor)))
        self.k = torch.nn.Parameter(torch.tensor(ICs_k, dtype=torch.float32).repeat(len(w_tensor)))
        
        # Store loss history for plotting
        self.loss_history = []
        
        # Track best parameters
        self.best_n = self.n.clone().detach()
        self.best_k = self.k.clone().detach()
        self.best_loss = float('inf')
    
    def forward(self, Physical_model):
        """
        Compute the theoretical transfer function given the current n, k values.
        """
        n_complex = self.n + 1j * self.k
        H_pred = Physical_model(n_complex, self.w_tensor, self.d)
        return H_pred
    
    def train_model(self, H_exp, Physical_model, loss_fn, 
                    optimizer=None, epochs=10000, lr=1e-4, verbose=True, updates=500, **loss_kwargs):
    
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute theoretical transfer function
            H_pred = self.forward(Physical_model)
            
            # Check if the loss function expects n and k for loss functions with smoothing terms
            loss_kwargs_dynamic = loss_kwargs.copy()  # Avoid modifying original kwargs
            if "n" in loss_fn.__code__.co_varnames and "k" in loss_fn.__code__.co_varnames:
                loss_kwargs_dynamic["n"] = self.n
                loss_kwargs_dynamic["k"] = self.k
            
            # Compute loss with additional parameters
            loss = loss_fn(H_exp, H_pred, **loss_kwargs_dynamic)
            self.loss_history.append(loss.item())
            
            # Store best parameters if loss improves
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_n = self.n.clone().detach()
                self.best_k = self.k.clone().detach()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % updates == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()} (Best Loss = {self.best_loss})")
        
        return self.loss_history

    


## Optimize for each frequency

class IndependentFrequencyModel:
    def __init__(self, freq, H_exp, n_real_init, n_imag_init, thickness, lr=0.001, max_iterations=500, tol=1e-5, patience=5):
        self.freq = torch.tensor(freq, dtype=torch.float32)
        self.H_exp = torch.tensor(H_exp, dtype=torch.complex64)

        # Ensure n_real_init is a tensor (convert single float to a tensor)
        if isinstance(n_real_init, (float, int)):  
            self.n_real_init = torch.tensor([n_real_init], dtype=torch.float32)
        else:
            self.n_real_init = torch.tensor(n_real_init, dtype=torch.float32)

        # Ensure n_imag_init is a tensor (same logic as above)
        if isinstance(n_imag_init, (float, int)):  
            self.n_imag_init = torch.tensor([n_imag_init], dtype=torch.float32)
        else:
            self.n_imag_init = torch.tensor(n_imag_init, dtype=torch.float32)

        # Store additional parameters
        self.lr = lr
        self.max_iterations = max_iterations
        self.tol = tol
        self.patience = patience
        self.L = thickness  # Sample thickness
        self.c = c  # Speed of light

    def H_th_function(self, n, freq):
        return (4 * n) / ((1 + n) ** 2) * torch.exp(-1j * self.L * 2 * torch.pi * freq * (n - 1) / self.c)

    def loss_function(self, H_proposed, H_exp):
        return (torch.real(H_proposed) - torch.real(H_exp)) ** 2 + (torch.imag(H_proposed) - torch.imag(H_exp)) ** 2

    def optimize(self):
        n_opt = torch.zeros(len(self.freq), dtype=torch.float32)
        k_opt = torch.zeros(len(self.freq), dtype=torch.float32)

        n_real_proposed = self.n_real_init[0].clone().detach().requires_grad_(True)
        n_imag_proposed = self.n_imag_init[0].clone().detach().requires_grad_(True)

        for i in range(len(self.freq)):
            f = self.freq[i]
            H_exp_f = self.H_exp[i]

            if i > 0:
                n_real_proposed = n_opt[i-1].clone().detach().requires_grad_(True)
                n_imag_proposed = k_opt[i-1].clone().detach().requires_grad_(True)

            optimizer = torch.optim.Adam([n_real_proposed, n_imag_proposed], lr=self.lr)
            best_loss = float("inf")
            patience_counter = 0

            for step in range(self.max_iterations):
                optimizer.zero_grad()
                n_complex = n_real_proposed + 1j * n_imag_proposed
                f_t = torch.tensor([f], dtype=torch.float32)

                H_pred = self.H_th_function(n_complex, f_t)
                loss = self.loss_function(H_pred, H_exp_f)
                loss.backward()
                optimizer.step()

                if abs(best_loss - loss.item()) < self.tol:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= self.patience:
                    break

                best_loss = loss.item()

            n_opt[i] = n_real_proposed.detach()
            k_opt[i] = n_imag_proposed.detach()

        return n_opt, k_opt
