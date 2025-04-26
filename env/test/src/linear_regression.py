import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def main():
    # Percorso del CSV generato
    csv_path = r'C:\Users\fabri\Desktop\my_repo\env\orders.csv'
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}. Assicurati che esista nella cartella corrente.")

    # 1) Carica dataset e conversione datetime
    df = pd.read_csv(csv_path, parse_dates=['order_datetime'])

    # 2) Aggrega su base giornaliera
    daily = df.set_index('order_datetime').resample('D').size().to_frame('num_orders')

    # 3) Feature engineering: lag e rolling mean
    daily['lag_1'] = daily['num_orders'].shift(1)
    daily['lag_7'] = daily['num_orders'].shift(7)
    daily['roll_mean_7'] = daily['num_orders'].shift(1).rolling(window=7).mean()

    # Aggiungi variabili calendario
    daily['day_of_week'] = daily.index.dayofweek
    daily['is_weekend'] = daily['day_of_week'].isin([5,6]).astype(int)
    daily['month'] = daily.index.month

    # 4) Pulizia: rimuovi NaN dovuti a lag
    daily = daily.dropna()

    # 5) Definisci train/test split (ultimi 60 giorni test)
    test_size = 60
    train = daily.iloc[:-test_size]
    test = daily.iloc[-test_size:]

    X_train = train[['lag_1', 'lag_7', 'roll_mean_7', 'day_of_week', 'is_weekend', 'month']]
    y_train = train['num_orders']
    X_test = test[['lag_1', 'lag_7', 'roll_mean_7', 'day_of_week', 'is_weekend', 'month']]
    y_test = test['num_orders']

    # 6) Baseline naive: predice il valore di lag_1
    y_pred_naive = X_test['lag_1']

    # 7) Regressione lineare
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_lin = model.predict(X_test)

    # 8) Calcolo metriche
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))

    print("Metriche su test set (ultimi 60 giorni):")
    print(f"- Baseline Naive: MAE = {mae_naive:.2f}, RMSE = {rmse_naive:.2f}")
    print(f"- Linear Regression: MAE = {mae_lin:.2f}, RMSE = {rmse_lin:.2f}\n")

    # 9) Plot comparativo
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, y_test, label='Reale', linewidth=2)
    plt.plot(test.index, y_pred_naive, label='Naive (lag_1)', linestyle='--')
    plt.plot(test.index, y_pred_lin, label='Linear Regression', linestyle=':')
    plt.title('Previsione ordini giornalieri: Reale vs Modelli')
    plt.xlabel('Data')
    plt.ylabel('Numero di ordini')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
