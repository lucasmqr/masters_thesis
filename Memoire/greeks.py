""""

Contient : 

- Code qui permet de tarcer les graphes sur les greeks 

à faire : tracer les greeks correctement avec la formule 
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import cmath

# --- Paramètres ---
K, T, r = 100, 1.0, 0.01
sigma_bs = 0.2
heston = {
    'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3,
    'rho': -0.7, 'v0': 0.04
}

# --- Black-Scholes Greeks ---
def bs_greeks(S):
    d1 = (np.log(S / K) + (r + 0.5 * sigma_bs ** 2) * T) / (sigma_bs * np.sqrt(T))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma_bs * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, vega

# --- Heston characteristic function ---
def char_fn(u, S):
    kappa, theta, sigma, rho, v0 = (heston[k] for k in ['kappa','theta','sigma','rho','v0'])
    x = np.log(S)
    a = kappa * theta
    d = np.sqrt((rho * sigma * 1j * u - kappa)**2 + (sigma**2) * (1j*u + u**2))
    g = (kappa - rho * sigma * 1j*u - d) / (kappa - rho * sigma * 1j*u + d)
    expDt = np.exp(-d * T)
    C = r * 1j * u * T + a / (sigma**2) * ((kappa - rho * sigma * 1j*u - d) * T - 2 * np.log((1 - g * expDt)/(1 - g)))
    D = (kappa - rho * sigma * 1j*u - d) / (sigma**2) * (1 - expDt)/(1 - g * expDt)
    return np.exp(C + D * v0 + 1j*u*x)

# --- Heston Grecque via intégrale ---
def heston_call_price(S):
    def integrand(u, j):
        num = np.exp(-1j*u*np.log(K)) * char_fn(u - (j-1)*1j, S)
        den = 1j*u * S**(j-1)
        return (num/den).real
    P1 = 0.5 + (1/np.pi) * quad(lambda u: integrand(u,1), 0, 100)[0]
    P2 = 0.5 + (1/np.pi) * quad(lambda u: integrand(u,2), 0, 100)[0]
    return S * P1 - K * np.exp(-r*T) * P2

def heston_greeks_fd(S, h=1e-3):
    C0 = heston_call_price(S)
    Cp = heston_call_price(S + h)
    Cm = heston_call_price(S - h)
    delta = (Cp - Cm) / (2*h)
    gamma = (Cp - 2*C0 + Cm) / h**2
    # Vega via bump vol-of-vol? here simple bump in v0
    orig = heston['v0']
    heston['v0'] = orig + 1e-3
    C_v = heston_call_price(S)
    vega = (C_v - C0) / 1e-3
    heston['v0'] = orig
    return delta, gamma, vega

# --- Calcul et tracés ---
S_grid = np.linspace(80, 120, 15)
bs = np.array([bs_greeks(S) for S in S_grid])
hest = np.array([heston_greeks_fd(S) for S in S_grid])

labels = ['Delta', 'Gamma', 'Vega']
plt.figure(figsize=(18,5))

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(S_grid, bs[:,i], 'k--', label=f'{labels[i]} BS')
    plt.plot(S_grid, hest[:,i], 'r-', label=f'{labels[i]} Heston')
    plt.xlabel('Spot S')
    plt.title(labels[i])
    plt.legend()
    plt.grid()

plt.suptitle('Comparaison des grecques : Black‑Scholes vs Heston (analytique)')
plt.tight_layout()
plt.show()
