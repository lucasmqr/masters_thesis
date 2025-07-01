import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import quad

# Paramètres de marché
r = 0.02  # taux sans risque

# Fonction caractéristique
def heston_cf(phi, S0, K, T, r, kappa, theta, sigma, rho, v0, Pnum):
    x = np.log(S0)
    a = kappa * theta
    u = 0.5 if Pnum == 1 else -0.5
    b = kappa - rho * sigma if Pnum == 1 else kappa
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)
    
    C = r * phi * 1j * T + a / sigma**2 * ((b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = ((b - rho * sigma * phi * 1j + d) / sigma**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    
    return np.exp(C + D * v0 + 1j * phi * x)

# Prix d’option (semi-fermé)
def heston_price_cf(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    def integrand(phi, Pnum):
        cf = heston_cf(phi, S0, K, T, r, kappa, theta, sigma, rho, v0, Pnum)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))
    
    P1 = 0.5 + (1/np.pi) * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
    P2 = 0.5 + (1/np.pi) * quad(lambda phi: integrand(phi, 2), 0, 100)[0]
    
    if option_type == 'call':
        return S0 * P1 - K * np.exp(-r * T) * P2
    else:
        return K * np.exp(-r * T) * (1 - P2) - S0 * (1 - P1)


def residuals(params, df, r=0.02):
    kappa, theta, sigma, rho, v0 = params
    res = []

    for _, row in df.iterrows():
        S0 = row['Spot']
        K = row["Prix d'exercice"]
        T = (pd.to_datetime(row['Maturité']) - pd.Timestamp.today()).days / 365.0
        option_type = row['Type'].lower()
        market_price = row['last_price']

        if T <= 0 or K <= 0 or market_price <= 0:
            continue

        model_price = heston_price_cf(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type)
        res.append(model_price - market_price)

    return np.array(res)

# Données
df_bdd = pd.read_excel(r"C:\Users\malu\Documents\Perso\Memoire\BDD.xlsx")
df_bdd['last_price'] = (df_bdd['Bid'] + df_bdd['Ask']) / 2

# Paramètres initiaux et bornes
init_params = [1.0, 0.04, 0.5, -0.5, 0.04]
bounds = ([0.001, 0.001, 0.001, -0.7, 0.001],
          [10, 1.0, 3.0, 0.7, 1.0])

# Calibration
result = least_squares(residuals, init_params, args=(df_bdd,), bounds=bounds, method='trf')

print("Calibration terminée.")
print(f"Paramètres calibrés :\n kappa={result.x[0]:.4f}, theta={result.x[1]:.4f}, "
      f"sigma={result.x[2]:.4f}, rho={result.x[3]:.4f}, v0={result.x[4]:.4f}")


# Calculer les prix modèles avec paramètres initiaux
df_bdd['model_price_initial'] = df_bdd.apply(
    lambda row: heston_price_cf(
        row['Spot'], 
        row["Prix d'exercice"], 
        (pd.to_datetime(row['Maturité']) - pd.Timestamp.today()).days / 365.0,
        r,
        *init_params,
        option_type=row['Type'].lower()
    ) if (pd.to_datetime(row['Maturité']) - pd.Timestamp.today()).days > 0 else np.nan,
    axis=1
)

# Calculer les prix modèles avec paramètres calibrés
df_bdd['model_price_calibrated'] = df_bdd.apply(
    lambda row: heston_price_cf(
        row['Spot'], 
        row["Prix d'exercice"], 
        (pd.to_datetime(row['Maturité']) - pd.Timestamp.today()).days / 365.0,
        r,
        *result.x,
        option_type=row['Type'].lower()
    ) if (pd.to_datetime(row['Maturité']) - pd.Timestamp.today()).days > 0 else np.nan,
    axis=1
)

# Affichage pour vérifier
print(df_bdd[['Type', "Prix d'exercice", 'Maturité', 'last_price', 'model_price_initial', 'model_price_calibrated']])
