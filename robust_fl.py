import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TwoLayerNN(nn.Module):
    def __init__(self, init_W, init_b, init_a):
        super().__init__()
        # weights = torch.randn(N, d)
        # weights /= torch.linalg.norm(weights, dim=1).reshape(-1, 1)
        self.W = nn.Parameter(init_W.clone())
        self.b = nn.Parameter(init_b.clone())
        self.a = nn.Parameter(init_a.clone())
        
    def forward(self, X):
        return F.relu(X @ self.W.T + self.b) @ self.a

@torch.no_grad()
def cost(model, X, y):
    return torch.mean((model(X) - y) ** 2).item()

def grad_params(W, theta, b, X, y, delta, sigma, sigma_prime):
    n = X.shape[0]
    yhat = pred(W, theta, b, X, delta, sigma)
    V_1 = 1 / np.sqrt(n) * sigma_prime(W @ (X + delta).T + b.reshape(-1,1)) * theta.reshape(-1, 1)
    V_2 = 1 / np.sqrt(n) * X * (yhat - y).reshape(-1,1)
    grad_theta = 1 / n * sigma(W @ (X + delta).T + b.reshape(-1,1)) @ (yhat - y)
    grad_W = V_1 @ V_2
    grad_b = V_1 @ (1 / np.sqrt(n) * (yhat - y))
    return grad_theta, grad_W, grad_b


def optimize_delta(model, X, y, epsilon, stepsize=0.1,num_iter=1000, return_costs=False,
                   verbose=False, use_signed_grad=True):
    model.requires_grad_(False)
    if num_iter == 0:
        return torch.zeros_like(X)
    delta = epsilon / (10 * np.sqrt(X.shape[1])) * torch.randn_like(X)
    delta_norm = torch.linalg.norm(delta, dim=1).reshape(-1,1)
    delta = torch.where(delta_norm < epsilon, delta, epsilon * delta / (delta_norm + 1e-8))
    costs = []
    for i in range(num_iter):
        if verbose and i % (num_iter // 20) == 0:
            with torch.no_grad():
                print(cost(model, X + delta, y))
        
        delta.requires_grad = True
        loss = torch.mean((model(X + delta) - y) ** 2)
        loss.backward()

        with torch.no_grad():
            if use_signed_grad:
                delta += stepsize * torch.sign(delta.grad)
            else:
                delta += stepsize * delta.grad
            delta_norm = torch.linalg.norm(delta, dim=1).reshape(-1,1)
            delta = torch.where(delta_norm < epsilon, delta, epsilon * delta / (delta_norm + 1e-8))

        if return_costs:
            with torch.no_grad():
                costs.append(cost(model, X + delta, y))
    if return_costs:
        return delta, costs
    else:
        return delta


def adv_loss(model, X, y, epsilon, train_first_layer, stepsize_adv=0.1, num_iter_adv=5):
    delta = optimize_delta(model, X, y, epsilon, stepsize=stepsize_adv, num_iter=num_iter_adv)
    delta.requires_grad = False
    model.requires_grad_(True)
    model.W.requires_grad = train_first_layer
    return torch.mean((model(X + delta) - y) ** 2)

def generate_data(batch_size, W_true, a_true, b_true, target_func):
    d = W_true.shape[1]
    X = torch.randn(batch_size, d, device=device)
    y = target_func(X @ W_true.T + b_true) @ a_true
    return X, y

def online_train(model, batch_size, epsilon, X_test, y_test, W_true, a_true, b_true,
                 target_func, train_first_layer, **kwargs):
    num_iter_train = kwargs.get('num_iter_train', 5000)
    stepsize_train = kwargs.get('stepsize_train', 0.001)
    num_iter_adv_train = kwargs.get('num_iter_adv_train', 5)
    stepsize_adv_train = kwargs.get('stepsize_adv_train', 0.1)
    num_iter_adv_test = kwargs.get('num_iter_adv_test', 20)
    stepsize_adv_test = kwargs.get('stepsize_adv_test', 0.1)
    test_points = kwargs.get('test_points', 100)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=stepsize_train)
    costs = []

    for i in range(num_iter_train):
        X, y = generate_data(batch_size, W_true, a_true, b_true, target_func)
        adv_loss(model, X, y, epsilon, train_first_layer, 
                 stepsize_adv=stepsize_adv_train, num_iter_adv=num_iter_adv_train).backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % (num_iter_train // test_points) == 0:
            delta = optimize_delta(model, X_test, y_test, epsilon, num_iter=num_iter_adv_test, stepsize=stepsize_adv_test)
            costs.append(cost(model, X_test + delta, y_test))

        if i % (num_iter_train // 20) == 0:
            print(costs[-1])
            
    return costs

