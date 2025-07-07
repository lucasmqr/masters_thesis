"""
Fonction qui montre l'influence des paramètres sur le prix d'une option avec le modèle de Heston

à faire : 
- appeler la fonction (formule fermée)
- combiener ce fichier avec celui qui fait la même chose avec la volatilité implicite

"""

import numpy as np
import matplotlib.pyplot as plt

# Fonction de simulation MC Heston (repris de ton code)
def heston_mc_price(
    S0, K, T, r,
    kappa, theta, sigma, rho, v0,
    option_type="call",
    n_paths=20000,
    n_steps=100
):
    dt = T / n_steps
    prices = np.full(n_paths, S0, dtype=np.float64)  # bien préciser float64
    variances = np.full(n_paths, v0, dtype=np.float64)

    for _ in range(n_steps):
        Z1 = np.random.normal(size=n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_paths)

        variances = np.maximum(
            variances + kappa * (theta - variances) * dt + sigma * np.sqrt(np.maximum(variances, 0)) * np.sqrt(dt) * Z2,
            0
        )
        prices *= np.exp((r - 0.5 * variances) * dt + np.sqrt(variances * dt) * Z1)

    if option_type == "call":
        payoff = np.maximum(prices - K, 0)
    else:
        payoff = np.maximum(K - prices, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price

# Paramètres fixes
S0 = 150
K = 150
T = 0.5
r = 0.02
option_type = "call"

# Valeurs de base des paramètres Heston
base_params = {
    'kappa': 2.0,
    'theta': 0.04,
    'sigma': 0.5,
    'rho': -0.7,
    'v0': 0.04
}

# Pour chaque paramètre, on varie autour de sa valeur de base
param_ranges = {
    'kappa': np.linspace(0.1, 5, 10),
    'theta': np.linspace(0.01, 0.2, 10),
    'sigma': np.linspace(0.1, 1.0, 10),
    'rho': np.linspace(-0.9, 0.9, 10),
    'v0': np.linspace(0.01, 0.2, 10),
}

# Trace un graphique par paramètre
for param_name, values in param_ranges.items():
    prices = []
    print(f"Calculs pour {param_name}...")
    for val in values:
        params = base_params.copy()
        params[param_name] = val
        price = heston_mc_price(
            S0, K, T, r,
            params['kappa'], params['theta'], params['sigma'], params['rho'], params['v0'],
            option_type=option_type,
            n_paths=20000,
            n_steps=100
        )
        prices.append(price)
    
    plt.figure(figsize=(8,5))
    plt.plot(values, prices, marker='o')
    plt.title(f"Impact de {param_name} sur le prix Heston MC")
    plt.xlabel(param_name)
    plt.ylabel("Prix option")
    plt.grid(True)
    plt.show()
