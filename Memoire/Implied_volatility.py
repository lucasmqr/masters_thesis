"""

Contient : 

- Le tracé des graphes qui montrent l'influences des paramètres sur la vol implicite

à faire : appeler la fonction avec la formule fermée

"""




import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# --- Black-Scholes Call Price ---
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# --- Implied Volatility from Option Price ---
def implied_volatility_mc(price, S, K, T, r):
    try:
        return brentq(lambda sigma: black_scholes_call_price(S, K, T, r, sigma) - price, 1e-6, 5.0)
    except:
        return np.nan

# --- Heston Monte Carlo Pricer ---
def heston_mc_price(S0, K, T, r, kappa, theta, sigma, rho, v0, N=20000, M=100):
    dt = T / M
    S = np.zeros((N, M+1))
    v = np.zeros((N, M+1))
    S[:,0] = S0
    v[:,0] = v0

    for t in range(1, M+1):
        z1 = np.random.normal(size=N)
        z2 = rho*z1 + np.sqrt(1 - rho**2)*np.random.normal(size=N)
        v[:,t] = np.abs(v[:,t-1] + kappa*(theta - v[:,t-1])*dt + sigma*np.sqrt(np.maximum(v[:,t-1], 0)*dt)*z2)
        S[:,t] = S[:,t-1]*np.exp((r - 0.5*v[:,t-1])*dt + np.sqrt(np.maximum(v[:,t-1], 0)*dt)*z1)

    payoff = np.maximum(S[:,-1] - K, 0)
    price = np.exp(-r*T)*np.mean(payoff)
    return price

# --- Paramètres fixes ---
S0 = 100
T = 1.0
r = 0.01
strike_list = [120, 130, 150]
base_params = {'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7, 'v0': 0.04}

# --- Paramètres à faire varier (10 valeurs chacun) ---
param_variations = {
    'kappa': np.linspace(0.5, 5.0, 10),
    'theta': np.linspace(0.01, 0.2, 10),
    'sigma': np.linspace(0.1, 1.0, 10),
    'rho': np.linspace(-0.9, 0.9, 10),
    'v0': np.linspace(0.01, 0.2, 10)
}

# --- Graphe pour chaque paramètre ---
for param_name, values in param_variations.items():
    plt.figure()
    for K in strike_list:
        ivs = []
        print(f"\nCalculs pour {param_name} avec K={K}")
        for val in values:
            params = base_params.copy()
            params[param_name] = val
            price = heston_mc_price(S0, K, T, r, **params)
            iv = implied_volatility_mc(price, S0, K, T, r)
            ivs.append(iv)
            print(f"{param_name}={val:.3f}, IV={iv:.4f}")
        plt.plot(values, ivs, marker='o', label=f"K={K}")

    plt.title(f"Influence de '{param_name}' sur l'IV (S0=100)")
    plt.xlabel(param_name)
    plt.ylabel("Volatilité implicite (MC via Heston)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.show()
