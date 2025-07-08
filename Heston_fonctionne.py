"""
Contient : 
- Fonction qui donne le prix avec le modèle de Heston (formule fermée)
- Code qui peremt de réalsier le tracé pour différents spots

"""



import numpy as np
import matplotlib.pyplot as plt

i = complex(0, 1)

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

def priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho):
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
prices = [priceHestonMid(S, K, r, T, sigma, kappa, theta, volvol, rho) for S in spot_range]

# Tracé
plt.plot(spot_range, prices, marker='o')
plt.xlabel('Spot S')
plt.ylabel('Prix Call Heston')
plt.title(f'Prix Call Heston en fonction du Spot (Strike={K})')
plt.grid(True)
plt.show()


## faire graphe qui montre si on retrouve un skew ou smile de vol 

def graph_des_grecques():
    return 0 