#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    csv_path = r'C:\Users\fabri\Desktop\my_repo\env\orders.csv'
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}. Assicurati che esista nella cartella corrente.")

    # 1) Carica dataset e conversione datetime
    df = pd.read_csv(csv_path, parse_dates=['order_datetime'])

    # 2) Aggrega su base giornaliera
    daily = df.set_index('order_datetime').resample('D').size().to_frame('num_orders')

    # 3) Feature engineering: lag e rolling
    daily['lag_1'] = daily['num_orders'].shift(1)
    daily['lag_7'] = daily['num_orders'].shift(7)
    daily['roll_mean_7'] = daily['num_orders'].shift(1).rolling(window=7).mean()

    # 4) Variabili temporali e esogene
    daily['day_of_week'] = daily.index.dayofweek
    daily['is_weekend'] = daily['day_of_week'].isin([5,6]).astype(int)
    # Se hai dati orari, puoi estrarre hour; per dati giornalieri usiamo placeholder 0
    daily['hour'] = 0
    daily['is_holiday'] = df.set_index('order_datetime').resample('D')['is_holiday'].max().fillna(0).astype(int)
    daily['season'] = df.set_index('order_datetime').resample('D')['season'].first().fillna('none')
    daily['holiday_name'] = df.set_index('order_datetime').resample('D')['holiday_name'].first().fillna('none')
    daily['distance_km'] = df.set_index('order_datetime').resample('D')['distance_km'].mean()
    daily['temperature_C'] = df.set_index('order_datetime').resample('D')['temperature_C'].mean()

    # 5) Pulizia dei NaN
    daily = daily.dropna()

    # 6) One-hot encoding categorie
    cat_cols = ['day_of_week', 'season', 'holiday_name']
    daily = pd.get_dummies(daily, columns=cat_cols, drop_first=True)

    # 7) Train/test split
    test_size = 60
    train = daily.iloc[:-test_size]
    test = daily.iloc[-test_size:]

    X_train = train.drop(columns=['num_orders'])
    y_train = train['num_orders']
    X_test = test.drop(columns=['num_orders'])
    y_test = test['num_orders']

    # 8) Modello GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        max_features=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 9) Metriche
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Metriche su test set (ultimi 60 giorni):")
    print(f"- GradientBoostingRegressor: MAE = {mae:.2f}, RMSE = {rmse:.2f}\n")

    # 10) Plot comparativo
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, y_test, label='Reale', linewidth=2)
    plt.plot(test.index, y_pred, label='GBR (sklearn)', linestyle='--')
    plt.title('Previsione ordini giornalieri con GradientBoostingRegressor')
    plt.xlabel('Data')
    plt.ylabel('Numero di ordini')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()