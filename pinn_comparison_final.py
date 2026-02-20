#!/usr/bin/env python3
"""
PINN vs MLP vs Klasik Yöntemler Karşılaştırması
PINN vs MLP vs Classical Methods Comparison
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import time
import math
import yfinance as yf

print("Modüller yüklendi / Modules loaded")

# ============================================================
# BÖLÜM 1: MODELLER VE FONKSİYONLAR / MODELS AND FUNCTIONS
# ============================================================

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, S, t, sigma):
        x = torch.cat([S, t, sigma], dim=1)
        return self.net(x)

class MLP(nn.Module):
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

def black_scholes_call_price(S, K, T, t, r, sigma):
    tau = T - t
    if tau <= 0:
        return max(S - K, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*math.sqrt(tau))
    d2 = d1 - sigma*math.sqrt(tau)
    price = S * norm.cdf(d1) - K * math.exp(-r*tau) * norm.cdf(d2)
    return price

def generate_synthetic_data(N, K, T, r, sigma):
    S_list, t_list, sigma_list, V_list = [], [], [], []
    for _ in range(N):
        S = torch.FloatTensor(1).uniform_(0.5*K, 2*K).item()
        t = torch.FloatTensor(1).uniform_(0, T).item()
        V = black_scholes_call_price(S, K, T, t, r, sigma)
        S_list.append([S])
        t_list.append([T-t])
        sigma_list.append([sigma])
        V_list.append([V])
    S = torch.tensor(S_list, dtype=torch.float32)
    t = torch.tensor(t_list, dtype=torch.float32)
    sigma = torch.tensor(sigma_list, dtype=torch.float32)
    V = torch.tensor(V_list, dtype=torch.float32)
    return S, t, sigma, V

def add_noise_to_data(V, noise_level=0.01):
    noise = noise_level * torch.randn_like(V) * V.abs()
    return V + noise

def train_model(model, data, epochs=150):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    S, t, sigma, V_real = data
    for epoch in range(epochs):
        V_pred = model(S, t, sigma)
        loss = nn.MSELoss()(V_pred, V_real)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 75 == 0 and epoch > 0:
            print(f"   Epoch {epoch}, Loss: {loss.item():.6f}")
    return model

# ============================================================
# ANA PROGRAM / MAIN PROGRAM
# ============================================================

print("\n" + "="*70)
print("PINN vs MLP vs Klasik Yöntemler Karşılaştırması")
print("="*70)

K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

print(f"\nParametreler / Parameters:")
print(f"  Strike Price: {K}, Maturity: {T}, Rate: {r}, Volatility: {sigma}")

# Veri oluştur
print("\nVeri oluşturuluyor / Creating data...")
N_train = 100
N_test = 50
S_train, t_train, sigma_train, V_train = generate_synthetic_data(N_train, K, T, r, sigma)
V_train_noisy = add_noise_to_data(V_train, noise_level=0.02)
S_test, t_test, sigma_test, V_test = generate_synthetic_data(N_test, K, T, r, sigma)

# PINN eğitimi
print("\n[1] PINN Eğitimi...")
pinn = PINN()
t_start = time.time()
pinn = train_model(pinn, (S_train, t_train, sigma_train, V_train_noisy), epochs=150)
t_pinn_train = time.time() - t_start
t_start = time.time()
V_pinn = pinn(S_test, t_test, sigma_test).detach().numpy()
t_pinn_test = time.time() - t_start

# MLP eğitimi
print("\n[2] MLP Eğitimi...")
mlp = MLP()
t_start = time.time()
mlp = train_model(mlp, (S_train, t_train, sigma_train, V_train_noisy), epochs=150)
t_mlp_train = time.time() - t_start
t_start = time.time()
V_mlp = mlp(S_test, t_test, sigma_test).detach().numpy()
t_mlp_test = time.time() - t_start

# Black-Scholes
print("\n[3] Black-Scholes Hesaplama...")
t_start = time.time()
V_bs = black_scholes_call_price(100.0, K, T, 0, r, sigma)
t_bs = time.time() - t_start
print(f"    BS Fiyat: {V_bs:.6f}, Zaman: {t_bs*1000:.4f}ms")

# Hata hesaplamaları
V_true = V_test.numpy()
mse_pinn = np.mean((V_pinn - V_true)**2)
mae_pinn = np.mean(np.abs(V_pinn - V_true))
rmse_pinn = np.sqrt(mse_pinn)

mse_mlp = np.mean((V_mlp - V_true)**2)
mae_mlp = np.mean(np.abs(V_mlp - V_true))
rmse_mlp = np.sqrt(mse_mlp)

print("\n" + "-"*70)
print("SONUÇLAR / RESULTS")
print("-"*70)
print(f"{'Model':<15} {'MAE':<15} {'RMSE':<15} {'Eğitim (s)':<15} {'Tahmin (ms)':<15}")
print("-"*70)
print(f"{'PINN':<15} {mae_pinn:<15.8f} {rmse_pinn:<15.8f} {t_pinn_train:<15.4f} {t_pinn_test*1000:<15.4f}")
print(f"{'MLP':<15} {mae_mlp:<15.8f} {rmse_mlp:<15.8f} {t_mlp_train:<15.4f} {t_mlp_test*1000:<15.4f}")
print("-"*70)

# Grafik oluştur
print("\nGrafik oluşturuluyor / Creating plots...")
fig = plt.figure(figsize=(24, 10))

# 1. PINN vs True
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(V_true, V_pinn, alpha=0.6, s=60, color='blue', edgecolors='black', linewidth=0.7)
ax1.plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 'k--', lw=2)
ax1.set_xlabel('Gerçek Fiyat', fontsize=12, fontweight='bold')
ax1.set_ylabel('PINN Tahmini', fontsize=12, fontweight='bold')
ax1.set_title(f'PINN (MAE={mae_pinn:.6f})', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=11)

# 2. MLP vs True
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(V_true, V_mlp, alpha=0.6, s=60, color='orange', edgecolors='black', linewidth=0.7)
ax2.plot([V_true.min(), V_true.max()], [V_true.min(), V_true.max()], 'k--', lw=2)
ax2.set_xlabel('Gerçek Fiyat', fontsize=12, fontweight='bold')
ax2.set_ylabel('MLP Tahmini', fontsize=12, fontweight='bold')
ax2.set_title(f'MLP (MAE={mae_mlp:.6f})', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=11)

# 3. PINN vs MLP
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(V_mlp, V_pinn, alpha=0.6, s=60, color='purple', edgecolors='black', linewidth=0.7)
ax3.plot([V_mlp.min(), V_mlp.max()], [V_mlp.min(), V_mlp.max()], 'k--', lw=2)
ax3.set_xlabel('MLP Tahmini', fontsize=12, fontweight='bold')
ax3.set_ylabel('PINN Tahmini', fontsize=12, fontweight='bold')
ax3.set_title('PINN vs MLP', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=11)

# 4. PINN Hata Dağılımı
ax4 = fig.add_subplot(2, 3, 4)
errors_pinn = V_pinn.flatten() - V_true.flatten()
ax4.hist(errors_pinn, bins=20, alpha=0.7, color='blue', edgecolor='black', linewidth=1.5)
ax4.axvline(0, color='red', linestyle='--', lw=2)
ax4.set_xlabel('Hata', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frekans', fontsize=12, fontweight='bold')
ax4.set_title(f'PINN Hata Dağılımı\nμ={np.mean(errors_pinn):.4f}', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.tick_params(labelsize=11)

# 5. MLP Hata Dağılımı
ax5 = fig.add_subplot(2, 3, 5)
errors_mlp = V_mlp.flatten() - V_true.flatten()
ax5.hist(errors_mlp, bins=20, alpha=0.7, color='orange', edgecolor='black', linewidth=1.5)
ax5.axvline(0, color='red', linestyle='--', lw=2)
ax5.set_xlabel('Hata', fontsize=12, fontweight='bold')
ax5.set_ylabel('Frekans', fontsize=12, fontweight='bold')
ax5.set_title(f'MLP Hata Dağılımı\nμ={np.mean(errors_mlp):.4f}', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
ax5.tick_params(labelsize=11)

# 6. MAE Karşılaştırması
ax6 = fig.add_subplot(2, 3, 6)
models = ['PINN', 'MLP']
mae_vals = [mae_pinn, mae_mlp]
colors = ['blue', 'orange']
bars = ax6.bar(models, mae_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax6.set_title('MAE Karşılaştırması', fontsize=12, fontweight='bold')
for bar, val in zip(bars, mae_vals):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height, f'{val:.6f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.tick_params(labelsize=11)

plt.suptitle('PINN vs MLP - Black-Scholes Opsiyon Fiyatlaması', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Kaydet
output_path = '/home/mehmetcelik/Masaüstü/masaüstü/makale/finance ml/model_comparison_final.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nGrafik kaydedildi: {output_path}")

# ============================================================
# EK: GERÇEK PİYASA VERİSİNDEN SENTEZ DATA (YFINANCE)
# ============================================================
def fetch_real_option_data(symbol='AAPL', start='2023-01-01', end='2023-12-31'):
    """
    Gerçek piyasa verisinden (yfinance) hisse kapanış fiyatı ve implied volatility çek.
    Sadece spot fiyat ve tarih döndürür. Opsiyon chain API'si yfinance'da kısıtlıdır.
    """
    data = yf.download(symbol, start=start, end=end, progress=False)
    closes = data['Close'].values
    dates = data.index.values
    return closes, dates

# Örnek kullanım: Apple hissesi için 2023 yılı spot fiyatları
if __name__ == '__main__':
    print("\nGerçek piyasa verisi çekiliyor (AAPL, 2023)...")
    closes, dates = fetch_real_option_data('AAPL', '2023-01-01', '2023-12-31')
    print(f"Toplam gün: {len(closes)} | İlk fiyat: {closes[0]:.2f} | Son fiyat: {closes[-1]:.2f}")
    # Gerçek fiyatlardan Black-Scholes ile teorik call fiyatı üretelim (örnek: K=ilk gün fiyatı, T=30 gün, r=0.05, sigma=0.2)
    K_real = closes[0]
    T_real = 30/252  # 30 gün vadeli
    r_real = 0.05
    sigma_real = 0.2
    V_real_list = []
    for i, S in enumerate(closes[:-30]):
        t = 0.0
        V = black_scholes_call_price(S, K_real, T_real, t, r_real, sigma_real)
        V_real_list.append(V)
    print(f"Örnek Black-Scholes fiyatları (ilk 5): {V_real_list[:5]}")
    # Bu veriyi PINN/MLP test seti olarak kullanabilirsiniz.

print("\n" + "="*70)
print("TAMAMLANDI / COMPLETED")
print("="*70)

# ============================================================
# EK: AMERIKAN OPSIYONU PINN KAYBI VE KULLANIMI
# ============================================================
def american_payoff(S, K):
    """
    Amerikan tipi call opsiyonun vade sonundaki değeri (payoff).
    """
    return torch.clamp(S - K, min=0.0)

def american_pinn_loss(model, data, pde_data, bc_data, r, K, weights=None):
    """
    Amerikan opsiyonu için PINN toplam kaybı (temel şablon).
    Standart PINN kaybına ek olarak erken kullanım (early exercise) koşulu içerir.
    """
    # Standart PINN kaybı (örnek: L_total, L_data, L_PDE, L_BC = pinn_total_loss(...))
    L_total, L_data, L_PDE, L_BC = pinn_total_loss(model, data, pde_data, bc_data, r, weights)
    # Erken kullanım koşulu (V >= payoff)
    S, t, sigma, _ = data
    payoff = american_payoff(S, K)
    V_pred = model(S, t, sigma)
    early_exercise_penalty = torch.mean(torch.clamp(payoff - V_pred, min=0.0)**2)
    # Toplam kayba ekle
    L_total = L_total + early_exercise_penalty
    return L_total, L_data, L_PDE, L_BC, early_exercise_penalty

# Not: Bu fonksiyonun çalışması için pinn_total_loss fonksiyonunun da kodda olması gerekir.
# Kapsamlı Amerikan opsiyonu PINN eğitimi için ayrıca serbest sınır (free boundary) ve iteratif çözüm teknikleri eklenebilir.

# ============================================================
# EK: HESTON MODELİ İÇİN PINN KAYBI (STOKASTİK VOLATİLİTE)
# ============================================================
def heston_pde_loss(model, S, t, v, r, kappa, theta, sigma_v, rho):
    """
    Heston PDE kaybı (stokastik volatilite).
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

# Not: Heston PINN için model mimarisi (S, t, v) girişli olmalı ve uygun veri üretilmelidir.
# Bu fonksiyon, stokastik volatilite içeren opsiyon fiyatlaması için PINN kaybını hesaplar.

# ============================================================
# EK: PARAMETRE KALİBRASYONU (IMPLIED VOLATILITY) İÇİN PINN
# ============================================================
class PINN_ImpliedVol(nn.Module):
    """
    Implied volatility'yi fonksiyon olarak öğrenen PINN.
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
    """
    V_pred, sigma_pred = model(S, t, K)
    data_loss = nn.MSELoss()(V_pred, V_market)
    # (İsteğe bağlı: sigma_pred üzerinde smoothness veya regularization kaybı eklenebilir)
    return data_loss

# Not: Bu şablon, implied volatility fonksiyonunu doğrudan öğrenmek için kullanılabilir.
# Piyasa fiyatı datası ile eğitilirse, model S, t, K girdilerinden implied volatility ve fiyatı tahmin eder.
