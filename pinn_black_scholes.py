# --- Parametre Kalibrasyonu (Implied Volatility) için PINN / PINN for Parameter Calibration (Implied Volatility) ---
# PINN, piyasa fiyatlarından implied volatility fonksiyonunu öğrenebilir.
# PINN can learn the implied volatility function from market prices.

class PINN_ImpliedVol(nn.Module):
    """
    Implied volatility'yi fonksiyon olarak öğrenen PINN.
    PINN that learns implied volatility as a function.
    """
    def __init__(self):
        super().__init__()
        self.vol_net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Softplus()  # Volatilite negatif olamaz
        )
        self.price_net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, S, t, K):
        # Implied volatility fonksiyonu S, t, K'ya bağlı
        sigma = self.vol_net(torch.cat([S, t], dim=1))
        x = torch.cat([S, t, sigma], dim=1)
        return self.price_net(x), sigma

def implied_vol_pinn_loss(model, S, t, K, V_market, r):
    """
    Implied volatility PINN kaybı: Model fiyatı ile piyasa fiyatı arasındaki fark.
    Implied volatility PINN loss: Difference between model price and market price.
    """
    V_pred, sigma_pred = model(S, t, K)
    data_loss = nn.MSELoss()(V_pred, V_market)
    # (İsteğe bağlı: sigma_pred üzerinde smoothness veya regularization kaybı eklenebilir)
    return data_loss
# --- Heston Modeli için PINN Şablonu / PINN Template for Heston Model ---
# Heston modelinde volatilite de stokastiktir ve ek bir PDE terimi vardır.
# In the Heston model, volatility is also stochastic and there is an additional PDE term.

def heston_pde_loss(model, S, t, v, r, kappa, theta, sigma_v, rho):
    """
    Heston PDE kaybı (stokastik volatilite).
    Heston PDE loss (stochastic volatility).
    """
    S.requires_grad_(True)
    t.requires_grad_(True)
    v.requires_grad_(True)
    V = model(S, t, v)
    V_t = autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_S = autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = autograd.grad(V_S, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_v = autograd.grad(V, v, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_vv = autograd.grad(V_v, v, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_Sv = autograd.grad(V_S, v, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    # Heston PDE
    pde = (V_t + 0.5*v*S**2*V_SS + rho*sigma_v*v*S*V_Sv + 0.5*sigma_v**2*v*V_vv
        + r*S*V_S + kappa*(theta-v)*V_v - r*V)
    return torch.mean(pde**2)
# --- Amerikan Opsiyonu için PINN Şablonu / PINN Template for American Option ---
# Amerikan tipi opsiyonlarda erken kullanım hakkı (early exercise) vardır ve bu bir serbest sınır (free boundary) problemidir.
# American options have early exercise feature, which is a free boundary problem.

def american_payoff(S, K):
    """
    Amerikan tipi call opsiyonun vade sonundaki değeri (payoff).
    Payoff of an American call option at maturity.
    """
    return torch.clamp(S - K, min=0.0)

# Amerikan opsiyonları için PINN kaybı, klasik Black-Scholes PDE'ye ek olarak erken kullanım koşulunu da içerir.
# For American options, PINN loss includes early exercise constraint in addition to the classical Black-Scholes PDE.

def american_pinn_loss(model, data, pde_data, bc_data, r, K, weights=None):
    """
    Amerikan opsiyonu için PINN toplam kaybı (temel şablon).
    Total PINN loss for American option (basic template).
    """
    # Standart PINN kaybı
    L_total, L_data, L_PDE, L_BC = pinn_total_loss(model, data, pde_data, bc_data, r, weights)
    # Erken kullanım koşulu (V >= payoff)
    S, t, sigma, _ = data
    payoff = american_payoff(S, K)
    V_pred = model(S, t, sigma)
    early_exercise_penalty = torch.mean(torch.clamp(payoff - V_pred, min=0.0)**2)
    # Toplam kayba ekle
    L_total = L_total + early_exercise_penalty
    return L_total, L_data, L_PDE, L_BC, early_exercise_penalty
# --- Gürültü Ekleme ve MLP Karşılaştırması / Noise Addition & MLP Comparison ---
def add_noise_to_data(V, noise_level=0.01):
    """
    Opsiyon fiyatlarına oransal gürültü ekler.
    Adds proportional noise to option prices.
    """
    noise = noise_level * torch.randn_like(V) * V.abs()
    return V + noise

class MLP(nn.Module):
    """
    Klasik çok katmanlı perceptron (MLP) ağı.
    Classic Multi-Layer Perceptron (MLP) network.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, S, t, sigma):
        x = torch.cat([S, t, sigma], dim=1)
        return self.net(x)

def train_mlp(model, data, epochs=1000):
    """
    Klasik MLP'yi sadece veri kaybı ile eğitir.
    Trains a classic MLP with only data loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    S, t, sigma, V_real = data
    for epoch in range(epochs):
        V_pred = model(S, t, sigma)
        loss = nn.MSELoss()(V_pred, V_real)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"[MLP] Epoch {epoch}, Loss: {loss.item():.6f}")
# --- Sentetik Veri Üretimi / Synthetic Data Generation ---
import math
from scipy.stats import norm
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Diğer Yöntemler / Alternative Methods ---

# 1. Monte Carlo Simülasyonu / Monte Carlo Simulation
def monte_carlo_option_price(S, K, T, r, sigma, num_simulations=10000, num_steps=252):
    """
    Monte Carlo yöntemiyle opsiyon fiyatı tahmin eder.
    Estimates option price using Monte Carlo simulation.
    Türkçe: Dayanak varlık fiyatlarını simüle eder ve payoff'ların ortalamasını alır.
    English: Simulates underlying asset prices and averages the payoffs.
    """
    dt = T / num_steps
    payoffs = []
    for _ in range(num_simulations):
        S_t = S
        for _ in range(num_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            S_t = S_t * np.exp((r - 0.5*sigma**2)*dt + sigma*dW)
        payoff = max(S_t - K, 0)
        payoffs.append(payoff)
    price = np.exp(-r*T) * np.mean(payoffs)
    return price

# 2. Finite Difference (FD) Yöntemi / Finite Difference Method
def finite_difference_option_price(S, K, T, r, sigma, S_max=None, num_S=100, num_t=100):
    """
    Finite Difference yöntemiyle Black-Scholes PDE'sini çözer.
    Solves Black-Scholes PDE using Finite Difference method.
    Türkçe: Açık şema (explicit scheme) kullanarak PDE'yi ızgarada çözer.
    English: Uses explicit scheme to solve PDE on a grid.
    """
    if S_max is None:
        S_max = 2 * K
    dS = S_max / num_S
    dt = T / num_t
    V = np.zeros((num_S+1, num_t+1))
    S_vals = np.linspace(0, S_max, num_S+1)
    # Sınır koşulu: Vade sonunda / Boundary: At maturity
    V[:, 0] = np.maximum(S_vals - K, 0)
    # Sınır koşulu: S=0'da / Boundary: At S=0
    V[0, :] = 0
    # Sınır koşulu: S=S_max'ta / Boundary: At S=S_max
    V[-1, :] = S_max - K*np.exp(-r*(T-dt*np.arange(num_t+1)))
    # Explicit şema / Explicit scheme
    for j in range(1, num_t+1):
        for i in range(1, num_S):
            V_SS = (V[i+1, j-1] - 2*V[i, j-1] + V[i-1, j-1]) / dS**2
            V_S = (V[i+1, j-1] - V[i-1, j-1]) / (2*dS)
            V[i, j] = (V[i, j-1] + 0.5*sigma**2*S_vals[i]**2*V_SS*dt 
                       + r*S_vals[i]*V_S*dt - r*V[i, j-1]*dt)
    # S'nin en yakın indeksini bulup fiyat döndür
    idx = np.argmin(np.abs(S_vals - S))
    return V[idx, -1]

def black_scholes_call_price(S, K, T, t, r, sigma):
    """
    Black-Scholes formülü ile Avrupa tipi call opsiyon fiyatı üretir.
    Generates European call option price using Black-Scholes formula.
    S: Dayanak varlık fiyatı / Underlying asset price
    K: Kullanım fiyatı / Strike price
    T: Vade sonu zamanı / Maturity time
    t: Şu anki zaman / Current time
    r: Risksiz faiz oranı / Risk-free rate
    sigma: Volatilite / Volatility
    """
    tau = T - t
    if tau <= 0:
        return max(S - K, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*math.sqrt(tau))
    d2 = d1 - sigma*math.sqrt(tau)
    price = S * norm.cdf(d1) - K * math.exp(-r*tau) * norm.cdf(d2)
    return price

def generate_synthetic_data(N, K, T, r, sigma):
    """
    Sentetik Black-Scholes veri seti üretir.
    Generates a synthetic Black-Scholes dataset.
    N: Veri sayısı / Number of data points
    """
    S_list, t_list, sigma_list, V_list = [], [], [], []
    for _ in range(N):
        S = torch.FloatTensor(1).uniform_(0.5*K, 2*K).item()
        t = torch.FloatTensor(1).uniform_(0, T).item()
        V = black_scholes_call_price(S, K, T, t, r, sigma)
        S_list.append([S])
        t_list.append([T-t])  # PINN'de genellikle kalan süre kullanılır
        sigma_list.append([sigma])
        V_list.append([V])
    S = torch.tensor(S_list, dtype=torch.float32)
    t = torch.tensor(t_list, dtype=torch.float32)
    sigma = torch.tensor(sigma_list, dtype=torch.float32)
    V = torch.tensor(V_list, dtype=torch.float32)
    return S, t, sigma, V
# --- Sınır Koşulları Fonksiyonları / Boundary Condition Functions ---

def european_call_payoff(S, K):
    """
    Avrupa tipi call opsiyonun vade sonundaki değeri (payoff).
    Payoff of a European call option at maturity.
    """
    return torch.clamp(S - K, min=0.0)

def boundary_conditions(S, t, sigma, K, T):
    """
    Sınır koşulları için teorik değerleri üretir.
    Generates theoretical values for boundary conditions.
    - Vade sonunda (t=0): payoff
    - S=0: V=0
    - S->∞: V ~ S - K*exp(-r*t)
    """
    # t=0 (maturity): V = payoff
    payoff = european_call_payoff(S, K)
    # S=0: V=0
    zero_boundary = torch.zeros_like(t)
    # S->∞: V ~ S - K*exp(-r*t) (yaklaşık)
    # Bu sınır için S büyük bir değer seçilebilir
    return payoff, zero_boundary
# PINN ile Black-Scholes Denklemi Çözümü (Gerçek Veriyle)
# Solution of Black-Scholes Equation with PINN (with Real Data)

"""
Bu kodda, Black-Scholes opsiyon fiyatlama denklemini Physics-Informed Neural Network (PINN) ile çözüyoruz.
Gerçek piyasa verisiyle çalışacak şekilde yapılandırılmıştır.

In this code, we solve the Black-Scholes option pricing equation using a Physics-Informed Neural Network (PINN).
It is structured to work with real market data.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd

# PINN ağı tanımı / PINN network definition
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, S, t, sigma):
        # S: Dayanak varlık fiyatı / Underlying asset price
        # t: Vade sonuna kalan süre / Time to maturity
        # sigma: Volatilite / Volatility
        x = torch.cat([S, t, sigma], dim=1)
        return self.net(x)

# Black-Scholes PDE kaybı / Black-Scholes PDE loss
# ∂V/∂t + 0.5*σ^2*S^2*∂²V/∂S² + r*S*∂V/∂S - r*V = 0


# Sınır koşulu kaybı / Boundary condition loss
def bc_loss(model, S_bc, t_bc, sigma_bc, V_bc):
    """
    Sınır koşullarında (ör. vade sonu, S=0, S->∞) model çıktısı ile teorik değer arasındaki fark.
    Difference between model output and theoretical value at boundary conditions (e.g. maturity, S=0, S->∞).
    """
    V_pred = model(S_bc, t_bc, sigma_bc)
    return nn.MSELoss()(V_pred, V_bc)

# Black-Scholes PDE kaybı / Black-Scholes PDE loss
def bs_pde_loss(model, S, t, sigma, r):
    """
    Modelin türevlerinin Black-Scholes PDE'ye ne kadar uyduğunu ölçer.
    Measures how well the model's derivatives satisfy the Black-Scholes PDE.
    """
    S.requires_grad_(True)
    t.requires_grad_(True)
    sigma.requires_grad_(True)
    V = model(S, t, sigma)
    V_t = autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_S = autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_SS = autograd.grad(V_S, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    pde = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V
    return torch.mean(pde**2)

# Toplam kayıp fonksiyonu / Total loss function
def pinn_total_loss(model, data, pde_data, bc_data, r, weights=None):
    """
    PINN toplam kaybı: L_total = ω_data*L_data + ω_PDE*L_PDE + ω_BC*L_BC
    Total PINN loss: L_total = ω_data*L_data + ω_PDE*L_PDE + ω_BC*L_BC
    """
    if weights is None:
        weights = {'data': 1.0, 'pde': 1.0, 'bc': 1.0}
    S, t, sigma, V_real = data
    S_pde, t_pde, sigma_pde = pde_data
    S_bc, t_bc, sigma_bc, V_bc = bc_data
    # Veri kaybı / Data loss
    L_data = nn.MSELoss()(model(S, t, sigma), V_real)
    # PDE kaybı / PDE loss
    L_PDE = bs_pde_loss(model, S_pde, t_pde, sigma_pde, r)
    # Sınır koşulu kaybı / Boundary condition loss
    L_BC = bc_loss(model, S_bc, t_bc, sigma_bc, V_bc)
    # Toplam kayıp / Total loss
    L_total = weights['data'] * L_data + weights['pde'] * L_PDE + weights['bc'] * L_BC
    return L_total, L_data, L_PDE, L_BC

# Eğitim fonksiyonu / Training function
# Gerçek veriyle eğitim / Training with real data

# PINN eğitim fonksiyonu / PINN training function
def train(model, data, pde_data, bc_data, r, epochs=1000, weights=None):
    """
    PINN modelini üçlü kayıp fonksiyonu ile eğitir.
    Trains the PINN model with triple loss function.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        loss, L_data, L_PDE, L_BC = pinn_total_loss(model, data, pde_data, bc_data, r, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Data: {L_data.item():.6f}, PDE: {L_PDE.item():.6f}, BC: {L_BC.item():.6f}")

# Gerçek veri örneği / Example with real data
# S, t, sigma, V_real torch tensorları olmalı / S, t, sigma, V_real should be torch tensors
# Aşağıda örnek veri yükleme kodu verilmiştir / Example data loading code is given below

def load_real_data():
    """
    Burada, gerçek piyasa verinizi yükleyin ve torch tensorlarına dönüştürün.
    Load your real market data here and convert to torch tensors.
    """
    # Örnek: CSV dosyasından veri okuma / Example: Reading data from CSV
    # import pandas as pd
    # df = pd.read_csv('option_data.csv')
    # S = torch.tensor(df['S'].values, dtype=torch.float32).view(-1,1)
    # t = torch.tensor(df['t'].values, dtype=torch.float32).view(-1,1)
    # sigma = torch.tensor(df['sigma'].values, dtype=torch.float32).view(-1,1)
    # V_real = torch.tensor(df['V'].values, dtype=torch.float32).view(-1,1)
    # return S, t, sigma, V_real
    pass  # Gerçek veriyle doldurulacak / To be filled with real data

# --- Model Karşılaştırması / Model Comparison ---

def compare_models(S_test, K, T, r, sigma, num_models=4):
    """
    Farklı yöntemlerin sonuçlarını ve performansını karşılaştırır.
    Compares results and performance of different methods.
    
    Karşılaştırılan Yöntemler / Compared Methods:
    1. PINN
    2. Klasik MLP
    3. Monte Carlo
    4. Finite Difference
    5. Black-Scholes (teorik)
    """
    results = {}
    
    # 1. Black-Scholes (Ground Truth) / Teorik Değer
    print("\n[1] Black-Scholes (Ground Truth)...")
    t_start = time.time()
    V_bs = black_scholes_call_price(S_test, K, T, 0, r, sigma)
    t_bs = time.time() - t_start
    results['Black-Scholes'] = {'price': V_bs, 'time': t_bs}
    print(f"    Fiyat / Price: {V_bs:.4f}, Zaman / Time: {t_bs*1000:.2f}ms")
    
    # 2. Monte Carlo / Monte Carlo Simülasyonu
    print("[2] Monte Carlo...")
    t_start = time.time()
    V_mc = monte_carlo_option_price(S_test, K, T, r, sigma, num_simulations=10000)
    t_mc = time.time() - t_start
    results['Monte Carlo'] = {'price': V_mc, 'time': t_mc}
    error_mc = abs(V_mc - V_bs) / V_bs * 100
    print(f"    Fiyat / Price: {V_mc:.4f}, Hata / Error: {error_mc:.4f}%, Zaman / Time: {t_mc*1000:.2f}ms")
    
    # 3. Finite Difference / Finite Difference Yöntemi
    print("[3] Finite Difference...")
    t_start = time.time()
    V_fd = finite_difference_option_price(S_test, K, T, r, sigma, num_S=100, num_t=100)
    t_fd = time.time() - t_start
    results['Finite Difference'] = {'price': V_fd, 'time': t_fd}
    error_fd = abs(V_fd - V_bs) / V_bs * 100
    print(f"    Fiyat / Price: {V_fd:.4f}, Hata / Error: {error_fd:.4f}%, Zaman / Time: {t_fd*1000:.2f}ms")
    
    return results

def benchmark_deep_learning_models(S_train, t_train, sigma_train, V_train, S_test, t_test, sigma_test, V_true, r, epochs=500):
    """
    PINN ve MLP'nin performansını karşılaştırır.
    Compares performance of PINN and MLP.
    """
    print("\n" + "="*60)
    print("DERİN ÖĞRENME MODELLERİ KARŞILAŞTIRMASI / DEEP LEARNING MODELS COMPARISON")
    print("="*60)
    
    # PINN eğitimi
    print("\n[1] PINN Eğitimi / Training...")
    pinn_model = PINN()
    t_start = time.time()
    # Basit eğitim (tam uygulama için pde_data ve bc_data gerekir)
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        V_pred = pinn_model(S_train, t_train, sigma_train)
        loss = nn.MSELoss()(V_pred, V_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 and epoch > 0:
            print(f"   Epoch {epoch}, Loss: {loss.item():.6f}")
    t_pinn_train = time.time() - t_start
    
    # PINN tahmini
    t_start = time.time()
    V_pinn_pred = pinn_model(S_test, t_test, sigma_test).detach().numpy()
    t_pinn_test = time.time() - t_start
    
    # MLP eğitimi
    print("\n[2] MLP Eğitimi / Training...")
    mlp_model = MLP()
    t_start = time.time()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        V_pred = mlp_model(S_train, t_train, sigma_train)
        loss = nn.MSELoss()(V_pred, V_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 and epoch > 0:
            print(f"   Epoch {epoch}, Loss: {loss.item():.6f}")
    t_mlp_train = time.time() - t_start
    
    # MLP tahmini
    t_start = time.time()
    V_mlp_pred = mlp_model(S_test, t_test, sigma_test).detach().numpy()
    t_mlp_test = time.time() - t_start
    
    # Hata hesaplamaları / Error calculations
    V_true_np = V_true.numpy()
    mse_pinn = np.mean((V_pinn_pred - V_true_np)**2)
    mae_pinn = np.mean(np.abs(V_pinn_pred - V_true_np))
    error_pinn = np.mean(np.abs(V_pinn_pred - V_true_np) / (np.abs(V_true_np) + 1e-8)) * 100
    
    mse_mlp = np.mean((V_mlp_pred - V_true_np)**2)
    mae_mlp = np.mean(np.abs(V_mlp_pred - V_true_np))
    error_mlp = np.mean(np.abs(V_mlp_pred - V_true_np) / (np.abs(V_true_np) + 1e-8)) * 100
    
    print("\n" + "-"*60)
    print("SONUÇLAR / RESULTS")
    print("-"*60)
    print(f"\n{'Metrik / Metric':<25} {'PINN':<15} {'MLP':<15}")
    print("-"*60)
    print(f"{'Eğitim Zamanı (s) / Training Time':<25} {t_pinn_train:<15.4f} {t_mlp_train:<15.4f}")
    print(f"{'Tahmin Zamanı (ms) / Prediction Time':<25} {t_pinn_test*1000:<15.4f} {t_mlp_test*1000:<15.4f}")
    print(f"{'MSE':<25} {mse_pinn:<15.6f} {mse_mlp:<15.6f}")
    print(f"{'MAE':<25} {mae_pinn:<15.6f} {mae_mlp:<15.6f}")
    print(f"{'Ortalama Hata % / Mean Error %':<25} {error_pinn:<15.4f} {error_mlp:<15.4f}")
    
    return {
        'pinn': {'pred': V_pinn_pred, 'mse': mse_pinn, 'mae': mae_pinn, 'error': error_pinn, 'time': t_pinn_test},
        'mlp': {'pred': V_mlp_pred, 'mse': mse_mlp, 'mae': mae_mlp, 'error': error_mlp, 'time': t_mlp_test}
    }

def plot_comparison(V_true, V_pinn, V_mlp, V_mc, V_fd, V_bs, save_path=None):
    """
    Tüm modellerin tahminlerini görselleştirir.
    Visualizes predictions from all models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Grafik 1: Tahminler vs Gerçek / Predictions vs True
    ax = axes[0, 0]
    ax.scatter(V_true, V_pinn, alpha=0.5, label='PINN', s=20)
    ax.scatter(V_true, V_mlp, alpha=0.5, label='MLP', s=20)
    ax.plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 'k--', label='Perfect')
    ax.set_xlabel('Gerçek Fiyat / True Price')
    ax.set_ylabel('Tahmin Fiyat / Predicted Price')
    ax.set_title('PINN vs MLP - Tahmin Doğruluğu / Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafik 2: Hata Dağılımı / Error Distribution
    ax = axes[0, 1]
    errors_pinn = V_pinn.flatten() - V_true.numpy().flatten()
    errors_mlp = V_mlp.flatten() - V_true.numpy().flatten()
    ax.hist(errors_pinn, bins=30, alpha=0.6, label='PINN', color='blue')
    ax.hist(errors_mlp, bins=30, alpha=0.6, label='MLP', color='orange')
    ax.set_xlabel('Hata / Error')
    ax.set_ylabel('Frekans / Frequency')
    ax.set_title('Hata Dağılımı / Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Grafik 3: Yöntemler Arası Karşılaştırma / Comparison of Methods
    ax = axes[1, 0]
    methods = ['PINN', 'MLP', 'Monte Carlo', 'FD', 'BS']
    mae_values = [
        np.mean(np.abs(V_pinn - V_true.numpy())),
        np.mean(np.abs(V_mlp - V_true.numpy())),
        abs(V_mc - V_bs),
        abs(V_fd - V_bs),
        0
    ]
    colors = ['blue', 'orange', 'green', 'red', 'black']
    ax.bar(methods, mae_values, color=colors, alpha=0.7)
    ax.set_ylabel('MAE')
    ax.set_title('Tüm Yöntemlerin MAE Karşılaştırması / MAE Comparison of All Methods')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Grafik 4: Hesaplama Zamanı / Computation Time
    ax = axes[1, 1]
    times = [1e-3, 1e-3, 0.5, 0.1, 1e-4]  # Örnek zamanlar
    ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel('Zaman (s) / Time (s)')
    ax.set_yscale('log')
    ax.set_title('Hesaplama Zamanı / Computation Time')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Örnek: Sentetik veri oluşturma ve model karşılaştırması
    print("="*60)
    print("PINN BLACK-SCHOLES: MODELLERİ KARŞILAŞTIRMA")
    print("PINN BLACK-SCHOLES: MODEL COMPARISON")
    print("="*60)
    
    # Parametreler / Parameters
    K = 100.0  # Kullanım fiyatı / Strike price
    T = 1.0    # Vade / Maturity
    r = 0.05   # Faiz oranı / Interest rate
    sigma = 0.2  # Volatilite / Volatility
    S_test = 100.0  # Test için dayanak fiyatı / Asset price for testing
    
    # Sentetik eğitim verisi oluştur / Generate synthetic training data
    print("\nSentetik veri üretiliyor / Generating synthetic data...")
    N_train = 100
    S_train, t_train, sigma_train, V_train = generate_synthetic_data(N_train, K, T, r, sigma)
    
    # Gürültülü veri / Noisy data
    V_train_noisy = add_noise_to_data(V_train, noise_level=0.02)
    
    # Test verisi / Test data
    N_test = 50
    S_test_data, t_test, sigma_test, V_test_true = generate_synthetic_data(N_test, K, T, r, sigma)
    
    # 1. Tek nokta karşılaştırması / Single point comparison
    print("\nTek Opsiyon için Yöntemler Karşılaştırması / Single Option Methods Comparison:")
    results = compare_models(S_test, K, T, r, sigma)
    
    # 2. Derin öğrenme modelleri karşılaştırması / Deep learning models comparison
    print("\nDerin Öğrenme Modellerinin Karşılaştırması / Deep Learning Models Comparison:")
    dl_results = benchmark_deep_learning_models(S_train, t_train, sigma_train, V_train_noisy,
                                                 S_test_data, t_test, sigma_test, V_test_true, r, epochs=300)
    
    # 3. Görselleştirme / Visualization
    print("\nGrafikler oluşturuluyor / Creating plots...")
    V_pinn = dl_results['pinn']['pred']
    V_mlp = dl_results['mlp']['pred']
    V_mc = results['Monte Carlo']['price']
    V_fd = results['Finite Difference']['price']
    V_bs = results['Black-Scholes']['price']
    plot_comparison(V_test_true, V_pinn, V_mlp, V_mc, V_fd, V_bs, 
                    save_path='/home/mehmetcelik/Masaüstü/masaüstü/makale/finance ml/model_comparison.png')
