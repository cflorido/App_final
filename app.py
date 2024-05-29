from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import adfuller
# Datos
import yfinance as yfin
import matplotlib.pyplot as plt
import os
from flask import request, url_for


# Gráficos
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import seaborn as sns

#analisis tecnico
import mplfinance as mpf

# Probabilidad y estadística
import math
from scipy.stats import norm, chi2, jarque_bera,shapiro
from scipy.optimize import brentq
from scipy import stats
from flask import Flask, render_template, request, jsonify
from io import BytesIO
import base64

# Importa Tkinter y otras bibliotecas necesarias para la generación de gráficos
import tkinter as tk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Define la aplicación Flask
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')



def app_function(strike, ciudad):   
    # Leer el archivo CSV
    filename = 'datos.csv'
    df= pd.read_csv(filename)

    # Lista de los nombres de las regiones
    regions = ['Anti', 'Narino', 'NortSan', 'Tolima', 'Cauca']

    # Itera sobre la lista de regiones
    for region in regions:
        max_col = f"Max_{region}"
        min_col = f"Min_{region}"
        tprom_col = f"TPROM_{region}"

        # Asegúrate de que las columnas Max y Min están en °C (si no es necesario, esta línea puede omitirse)
        df[max_col] = df[max_col]
        df[min_col] = df[min_col]

        # Calcula la temperatura promedio TPROM
        df[tprom_col] = (df[max_col] + df[min_col]) / 2

    # Seleccionar solo las columnas de temperatura promedio para cada región
    columns_to_plot = ['TPROM_Anti', 'TPROM_Narino', 'TPROM_NortSan', 'TPROM_Tolima', 'TPROM_Cauca']

    # Graficar las columnas seleccionadas
    plt.figure(figsize=(10, 5))
    for column in columns_to_plot:
        plt.plot(df.index, df[column], label=column)

    plt.legend()
    plt.title('Temperatura Promedio por Región')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura (°C)')


    def fill_missing_data(df, regions):
        df['Fecha'] = pd.to_datetime(df['Fecha'])  # Convertimos la columna 'Fecha' a datetime
        df = df.set_index('Fecha')  # Establecemos 'Fecha' como el índice
        df = df.resample('D').mean()  # Cambiamos la frecuencia a diaria y tomamos el promedio

        for region in regions:
            tprom_col = f"TPROM_{region}"
            if tprom_col in df.columns:
                df[tprom_col].interpolate(method='nearest', inplace=True)  # Interpolamos los valores de 'TPROM' para cada región

        df = df.reset_index()  # Restablecemos 'Fecha' como una columna normal
        return df

    df = fill_missing_data(df, regions)

    # Seleccionar solo las columnas de temperatura promedio para cada región
    columns_to_plot = ['TPROM_Anti', 'TPROM_Narino', 'TPROM_NortSan', 'TPROM_Tolima', 'TPROM_Cauca']

    # Graficar las columnas seleccionadas
    plt.figure(figsize=(10, 5))
    for column in columns_to_plot:
        plt.plot(df.index, df[column], label=column)

    plt.legend()
    plt.title('Temperatura Promedio por Región')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura (°C)')

    # Limitar los años en el eje x hasta el año 2005
    plt.xlim(df.index.min(), pd.Timestamp('2006-7-13'))


    # test for stationarity using the ADF test
    temp = ['TPROM_Anti', 'TPROM_Narino', 'TPROM_NortSan', 'TPROM_Tolima', 'TPROM_Cauca']
    df_Temp_f_columns = ['TPROM_AntiMR', 'TPROM_NarinoMR', 'TPROM_NortSanMR', 'TPROM_TolimaMR', 'TPROM_CaucaMR']

    # Create an empty DataFrame with the desired columns and 366 rows
    df_Temp_f = pd.DataFrame(index=np.arange(366), columns=df_Temp_f_columns)

    # Function to simulate paths
    def simulate_path(m, sigma, eta, dt, T, n, S0, sigma_e):
        timesteps = int(T/dt)
        paths = np.zeros((n, timesteps+1))
        paths[:,0] = S0

        for i in range(n):
            for j in range(1, timesteps+1):
                paths[i,j] = m*(1-np.exp(-eta)) + np.exp(-eta)*paths[i,j-1] + np.random.normal(loc=0, scale=sigma_e)

        return paths

    # Loop over each temperature variable
    for idx, i in enumerate(temp):
        # Calculate parameters for mean reversion
        result = adfuller(df[i], autolag='AIC')

        y = df[i] - df[i].shift(1)
        X = sm.add_constant(df[i].shift(1))
        model = sm.OLS(y, X, missing='drop')
        result = model.fit()

        a = result.params[0]
        b = result.params[1]
        residuals = result.resid
        sigma_e = np.std(residuals)
        sigma = np.std(df[i])
        m = -a/b
        eta = -np.log(1+b)

        N = 365
        T = 1
        dt = T/N    # Time step
        n = 5      # Number of simulations
        S0 = df[i].values[-1]

        paths = simulate_path(m, sigma, eta, dt, T, n, S0, sigma_e)

        # Plot paths
        timesteps = int(T/dt)
        t = np.linspace(0, T, timesteps+1)
        plt.figure(figsize=(10,6))
        for j in range(n):
            plt.plot(t, paths[j])
        plt.xlabel('Time (years)')
        plt.ylabel('T [°C]')
        plt.title('Simulacion de Temperatura para ' + i)

        # Calculate the mean of the simulated paths
        Tp = np.mean(paths, axis=0)

        # Assign the values of Tp to the corresponding column in df_Temp_f
        df_Temp_f[df_Temp_f_columns[idx]] = Tp

    def fourier_series(t, *a):
        t = np.asarray(t, dtype=float)
        ret = a[0] * np.ones_like(t)
        for i in range(1, len(a)//2 + 1):
            ret += a[2*i-1] * np.sin(2 * np.pi * i * t / 366) + a[2*i] * np.cos(2 * np.pi * i * t / 366)
        return ret

    def deterministic_seasonality(df, regions):
        results = {}
        for region in regions:
            tprom_col = f"TPROM_{region}"
            seasonal_fit_col = f"seasonal_fit_{region}"

            if tprom_col in df.columns:
                df['day_of_year'] = df.index.dayofyear
                temp_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[tprom_col, 'day_of_year'])
                initial_params = [1] * 7

                popt, _ = curve_fit(fourier_series, temp_df['day_of_year'], temp_df[tprom_col], initial_params)
                df[seasonal_fit_col] = fourier_series(df['day_of_year'], *popt)

                results[region] = popt

        df.drop(columns=['day_of_year'], inplace=True)
        return df, results

    def project_temperatures(df, regions, results):
        future_days = np.arange(1, 367)  # Cambiar a 367 para tener 366 días
        df_future = pd.DataFrame(index=future_days)

        for region in regions:
            seasonal_fit_col = f"seasonal_fit_{region}"
            proj_col = f"projection_{region}"
            params = results[region]

            df_future[proj_col] = fourier_series(future_days, *params)

        return df_future

    # Asegúrate de que tu DataFrame original tenga una columna de fechas llamada 'Fecha'
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df.set_index('Fecha', inplace=True)

    # Definir los nombres de las regiones
    regions = ['Anti', 'Narino', 'NortSan', 'Tolima', 'Cauca']

    # Ajustar la serie de Fourier a los datos históricos
    df, results = deterministic_seasonality(df, regions)

    # Proyectar las temperaturas para el próximo año
    df_future = project_temperatures(df, regions, results)

    # Mostrar el DataFrame con las proyecciones

    # Graficar las proyecciones
    plt.figure(figsize=(14, 7))
    for region in regions:
        proj_col = f"projection_{region}"
        plt.plot(df_future.index, df_future[proj_col], label=f'Proyección {region}')
    plt.xlabel('Día del Año')
    plt.ylabel('Temperatura [°C]')
    plt.title('Proyección de Temperaturas para el Próximo Año (366 días)')
    plt.legend()


    df_TF = pd.DataFrame(index=np.arange(366), columns=regions)

    for region in regions:
        df_TF[region]=(df_Temp_f[f"TPROM_{region}MR"]+df_future[f"projection_{region}"])/2

    regions = ['Anti', 'Narino', 'NortSan', 'Tolima', 'Cauca']
    start_year = df.index.year.min()  # Año mínimo de tus datos
    end_year = 2024  # Año máximo deseado

    for region in regions:
        plt.figure(figsize=(10, 5))
        plt.plot(df_TF.index, df_TF[region], label='Proyeccion Final (Fourier + Mean Reversion)')
        plt.plot(df_future.index, df_future[f"projection_{region}"], label="Proyeccion Fourier")
        plt.plot(df_Temp_f.index, df_Temp_f[f"TPROM_{region}MR"], label='Proyeccion Mean Reversioin')

        plt.legend()
        plt.title(f'Ajuste de Temperaturas en {region} usando Transformada de Fourier y Modelos Estocásticos')
        plt.xlabel('Fecha')
        plt.ylabel('Temperatura (°C)')

        # Limitar los años de estudio en el eje x
        #plt.xlim(pd.Timestamp(f'{start_year}-01-01'), pd.Timestamp(f'{end_year}-01-01'))

    # Crear listas vacías para almacenar los valores de HDD y CDD
    HDD_values = {region: [] for region in regions}
    CDD_values = {region: [] for region in regions}

    # Iterar sobre las filas del DataFrame para cada región
    for region in regions:
        c = np.mean(df_TF[region])
        for index, row in df_TF.iterrows():
            # Calcular el HDD y CDD para la fila actual y la región actual
            hdd = np.maximum(c - row[region], 0)
            cdd = np.maximum(0, row[region] - c)
            # Agregar el valor de HDD y CDD a las listas correspondientes
            HDD_values[region].append(hdd)
            CDD_values[region].append(cdd)

    # Agregar las listas de valores de HDD y CDD al DataFrame como nuevas columnas para cada región
    for region in regions:
        df_TF[f'HDD_{region}'] = HDD_values[region]
        df_TF[f'CDD_{region}'] = CDD_values[region]

    # Iterar sobre cada región
    for region in regions:
        # Configurar las dimensiones del gráfico para cada región
        plt.figure(figsize=(10, 5))

        # Graficar HDD para la región actual
        plt.plot(df_TF.index, df_TF[region], label=f'HDD ({region})', marker='o', linestyle='-')
        # Graficar CDD para la región actual
        #plt.plot(df.index, df[f'CDD_{region}'], label=f'CDD ({region})', marker='o', linestyle='-')

        # Agregar título y etiquetas a los ejes para cada región
        plt.title(f'Heating Degree Days (HDD) y Cooling Degree Days (CDD) en {region}')
        plt.xlabel('Fecha')
        plt.ylabel('Grados-Día')

        # Mostrar la leyenda para cada región
        plt.legend()

        # Mostrar la cuadrícula
        plt.grid(True)

    #print(df_TF["Narino"])

    date_range = pd.date_range(start='2024-01-01', end='2024-12-31')

    df_TF.index = date_range

    def option_pricing(df, strike_price, r, g, start_date, end_date):
        option_prices = {}

        # Iterar sobre cada región
        for region in regions:
            # Filtrar el DataFrame para el período de tiempo y la región específicos
            df_period = df[start_date:end_date]

            # Calcular la suma de HDD y CDD para la región y el período de tiempo
            hdd_sum = df_period[f'HDD_{region}'].sum()
            cdd_sum = df_period[f'CDD_{region}'].sum()

            # Calcular el tiempo en años entre start_date y end_date
            time_period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365

            # Calcular el precio de la opción de compra (call) basada en HDD para la región actual
            call_hdd = g * np.exp(-r * time_period) * max(0, hdd_sum - strike_price)

            # Calcular el precio de la opción de venta (put) basada en HDD para la región actual
            put_hdd = g * np.exp(-r * time_period) * max(0, strike_price - hdd_sum)

            # Calcular el precio de la opción de compra (call) basada en CDD para la región actual
            call_cdd = g * np.exp(-r * time_period) * max(0, cdd_sum - strike_price)

            # Calcular el precio de la opción de venta (put) basada en CDD para la región actual
            put_cdd = g * np.exp(-r * time_period) * max(0, strike_price - cdd_sum)

            # Guardar los precios de las opciones para la región actual
            option_prices[region] = {'call_hdd': call_hdd, 'put_hdd': put_hdd, 'call_cdd': call_cdd, 'put_cdd': put_cdd}

        return option_prices
####Intento con todos
# Parámetros de ejemplo

    strike_price = strike
    risk_free_rate = 0.0521487
    index_point_price = 25
    regions2 = ['Anti', 'Narino', 'NortSan', 'Tolima', 'Cauca']
    regions = []
    regions.append(ciudad)
    
    # Calcula opciones para cada mes de cada año
    year = 2024
    results = []
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')
    for date in dates:
        start_date = date.strftime('%Y-%m-01')
        end_date = (date + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

        option_prices = option_pricing(df_TF, strike_price, risk_free_rate, index_point_price, start_date, end_date)

        for region in regions:
            result = {
                'Year': year,
                'Month': date.strftime('%B'),
                'Region': region,
                'Call HDD': option_prices[region]['call_hdd'],
                'Put HDD': option_prices[region]['put_hdd'],
                'Call CDD': option_prices[region]['call_cdd'],
                'Put CDD': option_prices[region]['put_cdd']
            }
            results.append(result)

    df_pricing = pd.DataFrame(results)
    df_pricing_region = df_pricing[df_pricing['Region'] == ciudad]

    df_pricing_region.plot(x='Month', y=['Call HDD', 'Put HDD', 'Call CDD', 'Put CDD'], kind='line')
    plt.title(f"Precio de Opciones Climáticas en {ciudad}")
    plt.xlabel('Mes')
    plt.ylabel('Precio')

    fig = plt.gcf()
    plt.close()
    return fig


# Ejecuta la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)