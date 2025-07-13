"""
Ce fichier contient  : 

- Un pricer d'option basé sur le modèle de Heston (call et put) en utilisant la formule fermée
- Deux fonctions qui permettent de tracer le graph du prix d'un call (et put) en fonction du spot 
"""


import numpy as np
import matplotlib.pyplot as plt

i = complex(0, 1)

#pricing de l'option en utilisant le modèle de Heston
def fHeston(s, St, K, r, T, sigma, kappa, theta, volvol, rho):
    prod = rho * sigma * i * s
    d = np.sqrt((prod - kappa)**2 + (sigma**2) * (i * s + s**2))
    g = (kappa - prod - d) / (kappa - prod + d)
    
    exp1 = np.exp(np.log(St) * i * s) * np.exp(i * s * r * T)
    exp2 = 1 - g * np.exp(-d * T)
    exp3 = 1 - g
    power = -2 * theta * kappa / (sigma**2)
    
    mainExp1 = exp1 * (exp2 / exp3)**power
    
    exp4 = theta * kappa * T / (sigma**2)
    exp5 = volvol / (sigma**2)
    exp6 = (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    mainExp2 = np.exp(exp4 * (kappa - prod - d) + exp5 * (kappa - prod - d) * exp6)
    
    return mainExp1 * mainExp2

def call_priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho):
    P, iterations, maxNumber = 0, 1000, 100
    ds = maxNumber / iterations
    element1 = 0.5 * (St - K * np.exp(-r * T))
    
    for j in range(1, iterations):
        s1 = ds * (2 * j + 1) / 2
        s2 = s1 - i
        
        numerator1 = fHeston(s2, St, K, r, T, sigma, kappa, theta, volvol, rho)
        numerator2 = K * fHeston(s1, St, K, r, T, sigma, kappa, theta, volvol, rho)
        denominator = np.exp(np.log(K) * i * s1) * i * s1
        
        P += ds * (numerator1 - numerator2) / denominator
    
    element2 = P / np.pi
    
    return np.real(element1 + element2)

def put_priceHeston(St, K, r, T, sigma, kappa, theta, volvol, rho):

    # On utilise la put call parity
    put = call_priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho) - St + K*np.exp(-r*T)

    return put 

#Permet de tracer pour différents spot le prix de l'option
def trace_call_Heston():
    # Paramètres Heston fixes
    r = 0.01
    T = 1.0
    sigma = 0.3
    kappa = 2.0
    theta = 0.04
    volvol = 0.3
    rho = -0.7
    K = 100  # Strike fixé

    # Faire varier le spot autour de 60 à 140 par exemple
    spot_range = np.linspace(60, 140, 30)

    prices_call = [call_priceHestonMid(S, K, r, T, sigma, kappa, theta, volvol, rho) for S in spot_range]

    plt.plot(spot_range, prices_call, label='Call Heston', color='green')
    plt.xlabel('Spot S')
    plt.ylabel('Prix')
    plt.legend()
    plt.show()

def trace_put_heston():
    # Paramètres Heston fixes
    r = 0.02
    T = 0.002
    sigma = 0.3
    kappa = 1
    theta = 0.05
    volvol = 0.025
    rho = -0.5
    K = 240  # Strike fixé

    # Faire varier le spot autour de 60 à 140 par exemple
    spot_range = np.linspace(180, 300, 100)

    prices_put = [put_priceHeston(S, K, r, T, sigma, kappa, theta, volvol, rho) for S in spot_range]

    plt.plot(spot_range, prices_put, label='Put Heston', color='green')
    plt.xlabel('Spot S')
    plt.ylabel('Prix')
    plt.legend()
    plt.show()
    return 0
    

trace_call_Heston()
trace_put_heston()