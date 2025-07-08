"""

Contient : 

- Le tracé des graphes qui montrent l'influences des paramètres sur la vol implicite qui est fourni par le modèle de Hestion , attention ca n'est pas la vol observé sur le marché par les traders

à faire : appeler la fonction avec la formule fermée


est-ce pertinent de faire ca ? 

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import Heston_fonctionne as hf # Permet d'utiliser la fonction de pricing de Heston

# Renvoie le prix dans le cadre du modèle de Black Scholes
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Permet de récupérer la "volatilité implicite équivalente qu'on obtient avec le modèle de Heston"
def implied_volatility_mc(price, S, K, T, r):
    try:
        return brentq(lambda sigma: black_scholes_call_price(S, K, T, r, sigma) - price, 1e-6, 5.0)
    except:
        return np.nan

# Paramètres fixes
S = 100
T = 1.0
r = 0.01
strike_list = [110, 120, 130]
base_params = {'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7, 'volvol': 0.04}

#Paramètres de Heston à faire varier (10 valeurs chacun)
param_variations = {
    'kappa': np.linspace(0.5, 5.0, 10),
    'theta': np.linspace(0.01, 0.2, 10),
    'sigma': np.linspace(0.1, 1.0, 10),
    'rho': np.linspace(-0.9, 0.9, 10),
    'volvol': np.linspace(0.01, 0.2, 10)
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
            price = hf.priceHestonMid(S,K,r,T,params['sigma'],params['kappa'],params['theta'],params['volvol'],params['rho'])
            iv = implied_volatility_mc(price, S, K, T, r)
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
