"""
Ce fichier contient :
- Une fonction qui retourne les grecques dans le cadre du modèle de BS
- Calcul des grecques avec Heston en utlisant la méthode des différences finies
- Un tracé afin de visualiser les grecques de chaque modèle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import Heston_pricer as hp  # type: ignore # module Heston_pricer.py

# --- Paramètres globaux ---
K, T, r = 100, 1, 0.01
sigma_bs = 0.1  # volatilité Black-Scholes

# Paramètres du modèle de Heston
heston = {
    'kappa': 1.0,
    'theta': 0.025,
    'sigma': 0.3,    # σ: vol-of-vol
    'rho': -0.5,
    'v0': 0.025       # variance initiale
}

# --- Black-Scholes Greeks ---
def bs_greeks(S):
    d1 = (np.log(S / K) + (r + 0.5 * sigma_bs**2) * T) / (sigma_bs * np.sqrt(T))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma_bs * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, vega


# --- Appel du pricer Heston 
def heston_call_price(S):
    return hp.call_priceHestonMid(
        S, K, r, T,
        heston['sigma'],      # sigma
        heston['kappa'],
        heston['theta'],
        heston['v0'],      # volvol
        heston['rho']
    )

# Greeks de Heston par différences finies 
def heston_greeks_fd(S, h=1e-3):
    C0 = heston_call_price(S)
    Cp = heston_call_price(S + h)
    Cm = heston_call_price(S - h)

    delta = (Cp - Cm) / (2 * h)
    gamma = (Cp - 2 * C0 + Cm) / h**2

    # Vega via bump sur v0
    original_v0 = heston['v0']
    heston['v0'] = original_v0 + 1e-3
    C_vega = heston_call_price(S)
    vega = (C_vega - C0) / 1e-3
    heston['v0'] = original_v0  # reset

    return delta, gamma, vega

# Tracé comparatif 
S_grid = np.linspace(80, 120, 15)
bs = np.array([bs_greeks(S) for S in S_grid])
hest = np.array([heston_greeks_fd(S) for S in S_grid])

labels = ['Delta', 'Gamma', 'Vega']
plt.figure(figsize=(18, 5))

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(S_grid, bs[:, i], 'k--', label=f'{labels[i]} BS')
    plt.plot(S_grid, hest[:, i], 'r-', label=f'{labels[i]} Heston')
    plt.xlabel('Spot S')
    plt.title(labels[i])
    plt.legend()

plt.suptitle('Comparaison des Grecs : Black-Scholes vs Heston (via hp.call_price)')
plt.tight_layout()
plt.show()
