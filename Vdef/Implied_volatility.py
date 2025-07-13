"""
Ce fichier contient  :
- Une fonction qui renvoie le prix d'une option (BS modèle)
- Une fonction qui renvoie l'implied volatility
- Une fonction qui peremt de tracer l'influence des paramètres du modèle de Heston sur la volatilité implicite du modèle
- Une fonction qui permet de visualiser le smile/skew 
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import Heston_pricer as hf # Permet d'utiliser la fonction de pricing de Heston


# Renvoie le prix dans le cadre du modèle de Black Scholes
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# def black_scholes_put_price(S, K, T, r, sigma):
#     put_bs= black_scholes_call_price(S, K, T, r, sigma)-S+K*np.exp(-r*T)
#     return put_bs

# Permet de récupérer la "volatilité implicite équivalente qu'on obtient avec le modèle de Heston"
def call_implied_volatility_mc(price, S, K, T, r):
    try:
        return brentq(lambda sigma: black_scholes_call_price(S, K, T, r, sigma) - price, 1e-6, 5.0)
    except:
        return np.nan

# def put_implied_volatility_mc(price, S, K, T, r):
    try:
        return brentq(lambda sigma: black_scholes_put_price(S, K, T, r, sigma) - price, 1e-6, 5.0)
    except:
        return np.nan

# Permet de tracer l'influence des paramètres du modèle de Heston sur la vol implicite equivalente de l'option
def trace_impact():

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
                price = hf.call_priceHestonMid(S,K,r,T,params['sigma'],params['kappa'],params['theta'],params['volvol'],params['rho'])
                iv = call_implied_volatility_mc(price, S, K, T, r)
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

trace_impact()


## faire graphe qui montre si on retrouve un skew ou smile de vol 
# On fixe la maturité et on trace le prix pour différents call en fonction du strike 

def graph_smile():
    St = 100
    r = 0.03
    T = 1.0

    sigma = 0.7848
    kappa = 0.7253
    theta = 0.1084
    volvol = 0.0299
    rho = -0.7784

    #put_strikes = np.linspace(60, 140, 30)
    call_strikes = np.linspace(60, 140, 30)

    #put_prices = [hf.put_priceHeston(St, K, r, T, sigma, kappa, theta, volvol, rho) for K in put_strikes]
    #put_ivs = [put_implied_volatility_mc(price, St, K, T, r) for price, K in zip(put_prices, put_strikes)]

    call_prices = [hf.call_priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho) for K in call_strikes]
    call_ivs = [call_implied_volatility_mc(price, St, K, T, r) for price, K in zip(call_prices, call_strikes)]

    #plt.plot(put_strikes, put_ivs, label="Put Volatility Smile", color="blue")
    plt.plot(call_strikes, call_ivs, label="Call Volatility Smile", color="orange")
    plt.xlabel("Strike")
    plt.ylabel("Volatilité implicite")
    plt.title("Volatility Smile - Modèle de Heston")
    plt.legend()
    plt.grid(True)
    plt.show()

graph_smile()




