#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from dateutil.easter import easter
import math

# Parametri del dataset
year = 2024  # Anno di riferimento per la simulazione
total_orders = 500_000
output_csv = 'synthetic_orders.csv'

# Genera la lista di date dell'anno
date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')

# Calcola le festività italiane principali
fixed_holidays = {
    f'{year}-01-01': "Capodanno",
    f'{year}-01-06': "Epifania",
    f'{year}-04-25': "Festa della Liberazione",
    f'{year}-05-01': "Festa dei Lavoratori",
    f'{year}-06-02': "Festa della Repubblica",
    f'{year}-08-15': "Ferragosto",
    f'{year}-11-01': "Ognissanti",
    f'{year}-12-08': "Immacolata Concezione",
    f'{year}-12-25': "Natale",
    f'{year}-12-26': "Santo Stefano",
}
# Festività mobili: Pasqua e Lunedì di Pasqua
easter_date = easter(year)
fixed_holidays[easter_date.strftime('%Y-%m-%d')] = "Pasqua"

pasquetta = easter_date + timedelta(days=1)
fixed_holidays[pasquetta.strftime('%Y-%m-%d')] = "Pasquetta"

# Costruisci lista di date e pesi
dates = []
weights = []
for d in date_range:
    key = d.strftime('%Y-%m-%d')
    w = 1.0
    # weekend
    if d.weekday() >= 5:
        w = 3.0
    # festività
    if key in fixed_holidays:
        w = 5.0
    dates.append(d)
    weights.append(w)
# Normalizza i pesi
weights = np.array(weights)
prob_dates = weights / weights.sum()

# Segmenti orari per gli ordini (in ore)
time_segments = [
    (11, 14, 0.3),  # pranzo
    (18, 22, 0.6),  # cena
    (14, 18, 0.1),  # pomeriggio leggero
]
# Normalizza le probabilità dei segmenti
seg_probs = np.array([seg[2] for seg in time_segments])
seg_probs = seg_probs / seg_probs.sum()

# Funzione per campionare un orario all'interno dei segmenti
def sample_time():
    idx = np.random.choice(len(time_segments), p=seg_probs)
    start_h, end_h, _ = time_segments[idx]
    hour = np.random.uniform(start_h, end_h)
    minute = np.random.uniform(0, 60)
    sec = np.random.uniform(0, 60)
    return time(int(hour), int(minute), int(sec))

# Calcola temperatura basata sul giorno dell'anno (stagionalità sinusoidale)
def sample_temperature(d):
    # media annuale ~15°C, ampiezza ~10°C
    doy = d.timetuple().tm_yday
    temp = 15 + 10 * math.sin(2 * math.pi * (doy / 365.0))
    return round(temp + np.random.normal(0, 2), 1)

# Campionamento delle date per ogni ordine
chosen_dates = np.random.choice(dates, size=total_orders, p=prob_dates)

# Creazione del DataFrame
data = []
for od in chosen_dates:
    dt = datetime.combine(od, sample_time())
    key = od.strftime('%Y-%m-%d')
    is_holiday = key in fixed_holidays
    holiday_name = fixed_holidays.get(key, '')
    dow = od.weekday()
    is_weekend = 1 if dow >= 5 else 0
    # caratteristiche ordine
    num_pizzas = np.random.poisson(lam=2) + 1
    total_amount = round(num_pizzas * np.random.uniform(7, 10) + np.random.normal(0, 2), 2)
    distance_km = round(np.random.exponential(scale=3), 2)
    temp = sample_temperature(od)
    data.append({
        'order_datetime': dt,
        'day_of_week': dow,
        'is_weekend': is_weekend,
        'is_holiday': int(is_holiday),
        'holiday_name': holiday_name,
        'num_pizzas': num_pizzas,
        'total_amount': total_amount,
        'distance_km': distance_km,
        'temperature_C': temp,
    })

# Converti in DataFrame
df = pd.DataFrame(data)
# Aggiungi ID univoco ordine
df.insert(0, 'order_id', range(1, len(df) + 1))

# Scrivi su CSV
df.to_csv(output_csv, index=False)
print(f"CSV generato: {output_csv} con {len(df)} record.")
