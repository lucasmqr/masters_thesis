"""
Ce fichier contient  :
- Une fonction qui permet de visualiser l'influence des paramètres du modèle de Heston sur le prix de l'option

"""

import numpy as np
import matplotlib.pyplot as plt
import Heston_pricer as hf # type: ignore # Permet d'utiliser la fonction de pricing de Heston

def trace_impact_parametres():
    # Paramètres de base
    S = 100      # Prix spot
    K = 100      # Strike
    r = 0.03     # Taux d'intérêt
    T = 1.0      # Maturité

    # Paramètres Heston de base
    base_params = {
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'volvol': 0.3
    }

    # Ranges de variation pour chaque paramètre
    param_ranges = {
        'kappa': np.linspace(0.1, 5, 10),
        'theta': np.linspace(0.01, 0.2, 10),
        'sigma': np.linspace(0.1, 1.0, 10),
        'rho': np.linspace(-0.9, 0.9, 10),
        'volvol': np.linspace(0.01, 0.2, 10),
    }

    # Analyse de sensibilité
    for param_name, values in param_ranges.items():
        prices = []
        print(f"Calculs pour {param_name}...")
        for val in values:
            # Mettre à jour uniquement le paramètre étudié
            params = base_params.copy()
            params[param_name] = val
            
            # Appel de la fonction avec tous les bons arguments
            price = hf.call_priceHestonMid(
                S, K, r, T,
                params['sigma'],
                params['kappa'],
                params['theta'],
                params['volvol'],
                params['rho']
            )
            prices.append(price)
        
        # Affichage graphique
        plt.figure(figsize=(8,5))
        plt.plot(values, prices, marker='o')
        plt.title(f"Impact de {param_name} sur le prix Heston")
        plt.xlabel(param_name)
        plt.ylabel("Prix de l'option")
        plt.grid(True)
        plt.show()

trace_impact_parametres()