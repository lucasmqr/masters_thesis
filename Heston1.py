import numpy as np
import yfinance as yf
import pandas as pd

def get_sample_options_long_term(ticker, num_calls=7, num_puts=7,
                                 call_min_pct=0.10, call_max_pct=0.30,
                                 put_min_pct=0.10, put_max_pct=0.30,
                                 min_maturity_years=0.5):
    """
    Récupère des options avec maturité ≥ min_maturity_years (par défaut 6 mois).
    Les calls sont dans [+10%; +30%] au-dessus du spot, les puts dans [-30%; -10%] en-dessous.

    Args:
        ticker (str): symbole.
        num_calls (int): nombre calls.
        num_puts (int): nombre puts.
        call_min_pct (float): min % au-dessus spot calls.
        call_max_pct (float): max % au-dessus spot calls.
        put_min_pct (float): min % en-dessous spot puts.
        put_max_pct (float): max % en-dessous spot puts.
        min_maturity_years (float): maturité minimale en années.

    Returns:
        pd.DataFrame: options filtrées.
    """

    stock = yf.Ticker(ticker)
    spot = stock.history(period='1d')['Close'].iloc[-1]
    expirations = stock.options

    if len(expirations) == 0:
        raise ValueError(f"Aucune option disponible pour {ticker}")

    today = pd.Timestamp.today()
    # Convertir en datetime
    expirations_dates = pd.to_datetime(expirations)

    # Filtrer expirations pour maturité >= min_maturity_years
    expirations_long = [exp for exp, dt in zip(expirations, expirations_dates)
                        if (dt - today).days / 365.0 >= min_maturity_years]

    if len(expirations_long) == 0:
        raise ValueError(f"Aucune expiration avec maturité >= {min_maturity_years*12} mois")

    calls_collected = []
    puts_collected = []

    for exp in expirations_long:
        if len(calls_collected) >= num_calls and len(puts_collected) >= num_puts:
            break

        opt_chain = stock.option_chain(exp)
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()

        calls['option_type'] = 'call'
        puts['option_type'] = 'put'

        # Filtrer selon strikes
        calls = calls[(calls['strike'] >= spot * (1 + call_min_pct)) & (calls['strike'] <= spot * (1 + call_max_pct))]
        puts = puts[(puts['strike'] <= spot * (1 - put_min_pct)) & (puts['strike'] >= spot * (1 - put_max_pct))]

        calls = calls.sort_values('strike')
        puts = puts.sort_values('strike', ascending=False)

        calls_to_add = calls.head(num_calls - len(calls_collected))
        puts_to_add = puts.head(num_puts - len(puts_collected))

        calls_to_add['maturity'] = pd.to_datetime(exp)
        puts_to_add['maturity'] = pd.to_datetime(exp)

        calls_to_add['spot'] = spot
        puts_to_add['spot'] = spot

        calls_collected.append(calls_to_add)
        puts_collected.append(puts_to_add)

    df_calls = pd.concat(calls_collected, ignore_index=True)
    df_puts = pd.concat(puts_collected, ignore_index=True)
    df = pd.concat([df_calls, df_puts], ignore_index=True)

    df['maturity'] = (df['maturity'] - today).dt.days / 365.0
    df = df[(df['lastPrice'] > 0) & (df['maturity'] >= min_maturity_years) & (df['strike'] > 0)]
    df = df.rename(columns={'lastPrice': 'last_price'})
    df = df.reset_index(drop=True)

    # Vérification du nombre récupéré
    if len(df[df['option_type'] == 'call']) < num_calls:
        print(f"Attention: Seulement {len(df[df['option_type'] == 'call'])} calls récupérés")
    if len(df[df['option_type'] == 'put']) < num_puts:
        print(f"Attention: Seulement {len(df[df['option_type'] == 'put'])} puts récupérés")

    df = df[['strike', 'last_price', 'option_type', 'maturity','spot']].copy()

    return df


print(get_sample_options_long_term("AAPL"))