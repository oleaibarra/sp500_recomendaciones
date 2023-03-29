import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import yfinance as yf
from fredapi import Fred
import plotly.express as px

# Título del sidebar
st.sidebar.title('Menú')

# Opciones del sidebar
opcion = st.sidebar.radio('Seleccione una opción:',
                          ('Qué es el S&P 500',
                           'Análisis del S&P 500',
                           'Perfiles de riesgo y objetivos de inversión',
                           'Selección de combinación riesgo-objetivo',
                           'KPIs'))

# Dependiendo de la opción seleccionada, se muestra el contenido correspondiente
if opcion == 'Qué es el S&P 500':
    st.write('El S&P 500 es un índice bursátil que sigue el rendimiento de las 500 empresas más grandes que cotizan en la bolsa de valores de Estados Unidos, por lo que no es un activo individual que se pueda comprar o vender directamente. Sin embargo, se pueden utilizar instrumentos financieros como ETFs, futuros y opciones para tomar posiciones en el S&P 500 y seguir diferentes estrategias de inversión.')
    st.write('En general, las estrategias de inversión en el S&P 500 se pueden dividir en dos categorías: estrategias a largo plazo y estrategias a corto plazo.')
    st.subheader('¿Por qué seguir una estrategia de largo plazo?')
    st.write('Generalmente se requiere un mayor conocimiento y experiencia del mercado bursátil para las estrategias a corto plazo, ya que involucran operaciones más frecuentes y de corta duración. Las estrategias a corto plazo como el trading diario y la inversión en ETFs inversos pueden ser riesgosas y pueden llevar a pérdidas significativas si no se comprenden bien.')
    st.write('Para tener éxito en las estrategias a corto plazo, los inversores deben estar bien informados sobre el mercado y las noticias financieras, y también deben tener una buena comprensión de las técnicas de análisis técnico y fundamental. Es importante saber cómo leer gráficos de precios, interpretar indicadores técnicos y estar al tanto de los eventos que puedan afectar al mercado.')
    st.write('Además, las estrategias a corto plazo suelen requerir una dedicación más intensiva de tiempo y recursos que las estrategias a largo plazo. Los inversores que utilizan estrategias a corto plazo deben estar dispuestos a monitorear constantemente el mercado y actuar rápidamente ante los cambios de precio.')
    st.write('En general, las estrategias a corto plazo pueden ser más arriesgadas que las estrategias a largo plazo, y requieren una mayor comprensión y experiencia del mercado bursátil. Se recomienda que los inversores novatos comiencen con estrategias a largo plazo y se familiaricen con el mercado antes de aventurarse en las estrategias a corto plazo.')

elif opcion == 'Análisis del S&P 500':
    # Opciones del sidebar
    grafica = st.sidebar.radio('Seleccione una gráfica:',
                            ('S&P_500 2000-2023',
                            'PIB vs S&P_500',
                            'Inflación vs S&P_500',
                            'Tasa FED vs S&P_500',
                            'riesgo'
                            ))
    
    if grafica == 'S&P_500 2000-2023':
        
        st.subheader('S&P500 2000-2023')

            # Load the S&P 500 data
        sp500 = pd.read_csv('https://github.com/oleaibarra/sp500_recomendaciones/blob/main/sp500.csv', header=0, index_col='Date', parse_dates=True)

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the adjusted close price of the S&P 500 over time
        ax.plot(sp500.index, sp500['Adj Close'])

        # Set the title, x-label, and y-label
        ax.set_title('S&P 500 Index')
        ax.set_xlabel('Year')
        ax.set_ylabel('Adjusted Closing Price')

        # Add vertical lines to represent significant events
        events = {'COVID-19': ['2020-03-23'],
                'Burbuja inmobiliaria': ['2002-12-17'],
                'Lehman brothers quiebara': ['2008-09-15'],
                'Conflicto Rusia-Ucrania': ['2022-02-24']}

        for event, dates in events.items():
            for date in dates:
                date_num = mdates.datestr2num(date)
                ax.axvline(x=date_num, color='r', linestyle='--', linewidth=1)
                ax.annotate(event, xy=(date_num, sp500.loc[date, 'Adj Close']), xytext=(-20, 20), 
                            textcoords='offset points', ha='center', va='bottom', rotation=90, fontsize=8,
                            color='r', arrowprops=dict(facecolor='r', edgecolor='r', arrowstyle='-|>'))

        # Set the x-ticks to display only the year
        date_fmt = '%Y'
        date_locator = mdates.YearLocator()
        date_formatter = mdates.DateFormatter(date_fmt)
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Show the plot in Streamlit
        st.pyplot(fig)
    
    elif grafica == 'PIB vs S&P_500':
        
        st.subheader('PIB vs S&P_500')
            # set start and end dates
        start_date = datetime.datetime(2000, 1, 1)
        end_date = datetime.datetime(2022, 1, 1)

        # download the GDP data from FRED
        gdp = pd.read_csv('https://fred.stlouisfed.org/data/GDP.txt', sep='\s+', skiprows=30, header=None, index_col=0, parse_dates=True)
        gdp = gdp.loc['2000':]

        # resample the GDP data to quarterly frequency
        gdp_q = gdp.resample('Q').last()

        # Load the S&P 500 data
        sp500 = pd.read_csv('https://github.com/oleaibarra/sp500_recomendaciones/blob/main/sp500.csv', header=0, index_col='Date', parse_dates=True)

        # resample the S&P 500 data to monthly frequency
        sp500_adj_close = sp500['Adj Close'].resample('M').last()

        # plot the data on the same graph
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.set_title('USA GDP vs. S&P 500 Index ')
        ax1.plot(gdp_q.index, gdp_q, color='red')
        ax1.set_ylabel('GDP', color='red')

        ax2 = ax1.twinx()
        ax2.plot(sp500_adj_close.index, sp500_adj_close, color='blue')
        ax2.set_ylabel('S&P 500', color='blue')

        # display the plot in streamlit
        st.pyplot(fig)
        
    elif grafica == 'Inflación vs S&P_500':
        st.subheader('Inflación vs S&P_500')
        # Download S&P 500 data
        sp500_1960 = yf.download("^GSPC", start='1960-01-01')

        # Resample the S&P 500 data to annual frequency
        sp500_adj_close = sp500_1960['Adj Close'].resample('Y').last()

        # Load the inflation data from the CSV file
        inflation_df = pd.read_csv('https://github.com/oleaibarra/sp500_recomendaciones/blob/main/inflacion_USA.csv', header=0, index_col='year', parse_dates=True)

        inflation_factors = 1 + (inflation_df['inflation'] / 100)
        cumulative_inflation_factors = inflation_factors.cumprod()

        # Create a new DataFrame for cumulative inflation
        df_inflation = pd.DataFrame({
            'year': inflation_df.index,
            'cumulative_inflation': (cumulative_inflation_factors - 1) * 100
        })
        df_inflation.set_index('year', inplace=True)

        # Adjust the S&P 500 data for inflation
        cumulative_inflation_factors_aligned = cumulative_inflation_factors.reindex(sp500_adj_close.index, method='ffill')
        sp500_adj_close_infl_adj = sp500_adj_close.div(cumulative_inflation_factors_aligned.values, axis=0)

        # Create a new figure and axis object
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Add the first plot (Inflation)
        ax1.set_title('Inflación vs. S&P 500 Index (precios ajustados a la inflación)')
        ax1.plot(df_inflation.index, df_inflation['cumulative_inflation'], color='red')
        ax1.set_ylabel('Inflation', color='red')

        # Add the second plot (S&P 500 Inflation-adjusted)
        ax2 = ax1.twinx()
        ax2.plot(sp500_adj_close_infl_adj.index, sp500_adj_close_infl_adj, color='blue')
        ax2.set_ylabel('S&P 500 (Inflation-adjusted)', color='blue')

        # Add the third plot (S&P 500)
        ax3 = ax1.twinx()
        ax3.plot(sp500_adj_close.index, sp500_adj_close, color='green')
        ax3.set_ylabel('S&P 500', color='green')

        # Remove spines for the third axis
        ax3.spines.right.set_position(('axes', 1.2))
        ax3.spines.right.set_visible(True)
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.tick_right()

        # Show the plot in Streamlit
        st.pyplot(fig)
        
    elif grafica == 'Tasa FED vs S&P_500':
        st.subheader('Tasa FED vs S&P_500')
        # Replace YOUR_API_KEY with your actual FRED API key
        fred = Fred(api_key='083c28f1f599f240f0439dd8205695e8')

        # Load the S&P 500 data
        sp500 = pd.read_csv('https://github.com/oleaibarra/sp500_recomendaciones/blob/main/sp500.csv', header=0, index_col='Date', parse_dates=True)

        # resample the S&P 500 data to monthly frequency
        sp500_adj_close = sp500['Adj Close'].resample('Q').last()

        # Get the unemployment rate data
        fed = fred.get_series('FEDFUNDS')
        fed = fed.loc['2000':]

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_title('Tasa FED vs. S&P 500 Index ')
        ax1.plot(fed.index, fed, color='red')
        ax1.set_ylabel('FEDFUNDS RATE', color='red')

        ax2 = ax1.twinx()
        ax2.plot(sp500_adj_close.index, sp500_adj_close, color='blue')
        ax2.set_ylabel('S&P 500', color='blue')

        # Show the plot in Streamlit
        st.pyplot(fig)
        
    elif grafica == 'riesgo':
        st.subheader('Rendimiento y riesgo de empresas del S&P 500')
        
        # Definimos una lista con los tickers de las empresas que queremos analizar
        tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']

        # Descargamos los datos de los precios de cierre ajustados para los últimos 5 años
        data = yf.download(tickers, start='2018-01-01', end='2023-03-25')['Adj Close']

        # Calculamos los rendimientos diarios de las empresas y del S&P 500
        returns = data.pct_change().dropna()
        sp500 = yf.download('^GSPC', start='2018-01-01', end='2023-03-25')['Adj Close'].pct_change().dropna()

        # Calculamos estadísticas de rendimiento y riesgo para las empresas y el S&P 500
        mean_returns = returns.mean()
        std_returns = returns.std()
        mean_sp500 = sp500.mean()
        std_sp500 = sp500.std()

        # Creamos un dataframe con los datos de rendimiento y riesgo de cada empresa y del S&P 500
        df = pd.DataFrame({'Empresa': tickers + ['S&P 500'],
                        'Rendimiento medio diario': mean_returns.values.tolist() + [mean_sp500],
                        'Volatilidad diaria': std_returns.values.tolist() + [std_sp500]})

        # Generamos un gráfico de dispersión para comparar las empresas con el S&P 500 en términos de rendimiento y riesgo
        fig = px.scatter(df, x='Volatilidad diaria', y='Rendimiento medio diario', color='Empresa')

        # Agregamos una línea que representa el rendimiento y el riesgo del S&P 500
        fig.add_shape(type='line', x0=0, y0=mean_sp500, x1=std_sp500*np.sqrt(252), y1=mean_sp500, line=dict(color='gainsboro', dash='dash'))

        # Agregamos texto para identificar la línea del S&P 500
        fig.add_annotation(x=std_sp500*np.sqrt(252), y=mean_sp500, text='S&P 500', showarrow=False, font=dict(size=14))

        # Agregamos título y etiquetas de los ejes
        fig.update_layout(title='Rendimiento y riesgo de empresas del S&P 500 vs S&P 500',
                        xaxis_title='Volatilidad diaria',
                        yaxis_title='Rendimiento medio diario')

        # Limitamos la gráfica en el eje x hasta 0.05
        fig.update_xaxes(range=[0,0.05])

        # Mostramos el gráfico en el dashboard de Streamlit
        st.plotly_chart(fig)
            
    
    
elif opcion == 'Perfiles de riesgo y objetivos de inversión':
    st.subheader('Perfiles de riesgo')
    st.write('Conservador: Este perfil busca minimizar el riesgo y preservar el capital, generalmente invirtiendo en instrumentos de renta fija y depósitos a plazo.')
    st.write('Moderado: Este perfil busca un equilibrio entre la preservación del capital y el crecimiento, invirtiendo en una combinación de instrumentos de renta fija y variable.')
    st.write('Agresivo: Este perfil busca maximizar el crecimiento de la inversión, asumiendo mayores riesgos en instrumentos de renta variable y activos de mayor riesgo.')
    st.write('Especulativo: Este perfil asume altos riesgos con el objetivo de obtener ganancias a corto plazo, invirtiendo en instrumentos de alta volatilidad y en mercados de nicho.')
    
    st.subheader('Objetivo de inversión')
    st.write('Preservación de capital: El objetivo es proteger el capital invertido y minimizar el riesgo de pérdidas.')
    st.write('Ingreso: El objetivo es generar ingresos regulares a través de los dividendos, intereses y otros pagos.')
    st.write('Crecimiento: El objetivo es aumentar el valor de la inversión a largo plazo mediante la apreciación del capital.')
        
    
elif opcion == 'Selección de combinación riesgo-objetivo':

    # Definimos las opciones para el selection box de perfiles
    perfiles = ['Conservador', 'Moderado', 'Agresivo', 'Especulativo']

    # Creamos el selection box para que el usuario seleccione su perfil de riesgo
    perfil_seleccionado = st.selectbox('Seleccione su perfil de riesgo:', perfiles)

    # Dependiendo del perfil seleccionado, habilitamos distintas opciones de objetivos de inversión
    if perfil_seleccionado == 'Conservador':
        objetivos = ['Preservación de capital', 'Ingreso']
    elif perfil_seleccionado == 'Moderado':
        objetivos = ['Ingreso', 'Crecimiento']
    else:
        objetivos = ['Ingreso', 'Crecimiento']

    # Creamos el selection box para que el usuario seleccione su objetivo de inversión
    objetivo_seleccionado = st.selectbox('Seleccione su objetivo de inversión:', objetivos)

    # Mostramos el perfil y objetivo seleccionados
    st.write(f'Perfil seleccionado: {perfil_seleccionado}')
    st.write(f'Objetivo seleccionado: {objetivo_seleccionado}')

    ###############################


    df = pd.read_parquet('https://github.com/oleaibarra/sp500_recomendaciones/blob/main/df_metrics_all.parquet')
    tickers = np.array(df.index).tolist()
    daily_volume = df['daily_volume'].to_dict()
    market_cap = df['market_cap'].to_dict()
    avg_daily_returns = df['avg_daily_returns'].to_dict()
    avg_anual_returns = df['avg_anual_returns'].to_dict()
    avg_daily_volatility = df['avg_daily_volatility'].to_dict()
    avg_anual_volatility = df['avg_anual_volatility'].to_dict()
    sharpe_ratio = df['sharpe_ratio'].to_dict()
    trynor_ratio = df['trynor_ratio'].to_dict()
    information_ratio= df['information_ratio'].to_dict()
    beta = df['beta'].to_dict()
    dia_invertir = df['dia_invertir'].to_dict()

    # Parámetros para recomendar empresas según perfil y objetivo

    if (perfil_seleccionado == 'Conservador') and (objetivo_seleccionado == 'Preservación de capital'):
        

        # 1.1 Perfil conservador con objetivo preservación de capital

        # Definir los umbrales y rangos para las métricas
        volatility_daily_threshold = 0.02
        volatility_annual_threshold = 0.2
        beta_threshold = 1.0
        returns_range = (0.0427, 0.06)

        # Filtrar las compañías que cumplan con las condiciones definidas
        conservative_companies = []
        for ticker in tickers:
            if (
                avg_daily_volatility[ticker] <= volatility_daily_threshold
                and avg_anual_volatility[ticker] <= volatility_annual_threshold
                and beta[ticker] <= beta_threshold
                and avg_anual_returns[ticker] >= returns_range[0]
                and avg_anual_returns[ticker] <= returns_range[1]
            ):
                conservative_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not conservative_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
            st.write('Considere invertir en bonos a 10 años; sin riesgo y con un interés anual de 3.77%.')
        else:
            for ticker in conservative_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')


    elif (perfil_seleccionado == 'Conservador') and (objetivo_seleccionado == 'Ingreso'):
        # Definir los umbrales y rangos para las métricas
        daily_volatility_threshold = 0.01
        annual_volatility_threshold = 0.15
        beta_threshold = 1.0
        annual_returns_range = (0.0427, 0.06)
        daily_returns_range = (0, 0.005)
        market_cap_threshold = 500000000
        daily_volume_threshold = 1000000

        # Filtrar las compañías que cumplan con las condiciones definidas
        conservative_income_companies = []
        for ticker in tickers:
            if (
                avg_daily_volatility[ticker] <= daily_volatility_threshold
                and avg_anual_volatility[ticker] <= annual_volatility_threshold
                and beta[ticker] <= beta_threshold
                and annual_returns_range[0] <= avg_anual_returns[ticker] <= annual_returns_range[1]
                and daily_returns_range[0] <= avg_daily_returns[ticker] <= daily_returns_range[1]
                and market_cap[ticker] >= market_cap_threshold
                and daily_volume[ticker] >= daily_volume_threshold
            ):
                conservative_income_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not conservative_income_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
            st.write('Considere invertir en bonos a 10 años; sin riesgo y con un interés anual de 3.77%.')
        else:
            st.write(f'Compañías recomendadas para el perfil {perfil_seleccionado} con objetivo {objetivo_seleccionado}:')
            for ticker in conservative_income_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')



    elif (perfil_seleccionado == 'Moderado') and (objetivo_seleccionado == 'Ingreso'):
        # Definir los umbrales y rangos para las métricas
        annual_returns_range = (0.06, 0.12)
        annual_volatility_threshold = 0.22
        daily_volatility_threshold = 0.015
        beta_threshold = 1
        daily_volume_threshold = 1000000

        # Filtrar las compañías que cumplan con las condiciones definidas
        moderate_income_companies = []
        for ticker in tickers:
            if (
                avg_anual_volatility[ticker] <= annual_volatility_threshold
                and avg_daily_volatility[ticker] <= daily_volatility_threshold
                and beta[ticker] <= beta_threshold
                and avg_anual_returns[ticker] >= annual_returns_range[0]
                and avg_anual_returns[ticker] <= annual_returns_range[1]
                and daily_volume[ticker] >= daily_volume_threshold
            ):
                moderate_income_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not moderate_income_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
        else:
            st.write(f'Compañías recomendadas para el perfil {perfil_seleccionado} con objetivo {objetivo_seleccionado}:')
            for ticker in moderate_income_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')


    elif (perfil_seleccionado == 'Moderado') and (objetivo_seleccionado == 'Crecimiento'):
        # Definir los umbrales y rangos para las métricas
        annual_returns_range = (0.1, 0.2)
        annual_volatility_threshold = 0.23
        daily_volatility_threshold = 0.02
        beta_threshold = 1.1
        market_cap_threshold = 10000000000
        daily_volume_threshold = 4000000

        # Filtrar las compañías que cumplan con las condiciones definidas
        moderate_growth_companies = []
        for ticker in tickers:
            if (
                avg_anual_volatility[ticker] <= annual_volatility_threshold
                and avg_daily_volatility[ticker] <= daily_volatility_threshold
                and beta[ticker] <= beta_threshold
                and avg_anual_returns[ticker] >= annual_returns_range[0]
                and avg_anual_returns[ticker] <= annual_returns_range[1]
                and market_cap[ticker] >= market_cap_threshold
                and daily_volume[ticker] >= daily_volume_threshold
            ):
                moderate_growth_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not moderate_growth_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
        else:
            st.write(f'Compañías recomendadas para el perfil {perfil_seleccionado} con objetivo {objetivo_seleccionado}:')
            for ticker in moderate_growth_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')


    elif (perfil_seleccionado == 'Agresivo') and (objetivo_seleccionado == 'Ingreso'):
        # Definir los umbrales y rangos para las métricas
        annual_returns_range = (0.15, 0.18)
        annual_volatility_threshold = 0.3
        daily_volatility_threshold = 0.02
        beta_threshold = 1.2
        market_cap_threshold = 20000000000
        daily_volume_threshold = 4000000

        # Filtrar las compañías que cumplan con las condiciones definidas
        aggressive_income_companies = []
        for ticker in tickers:
            if (
                avg_anual_volatility[ticker] <= annual_volatility_threshold
                and avg_daily_volatility[ticker] <= daily_volatility_threshold
                and beta[ticker] <= beta_threshold
                and avg_anual_returns[ticker] >= annual_returns_range[0]
                and avg_anual_returns[ticker] <= annual_returns_range[1]
                and market_cap[ticker] >= market_cap_threshold
                and daily_volume[ticker] >= daily_volume_threshold
            ):
                aggressive_income_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not aggressive_income_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
        else:
            st.write(f'Compañías recomendadas para el perfil {perfil_seleccionado} con objetivo {objetivo_seleccionado}:')
            for ticker in aggressive_income_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')


    elif (perfil_seleccionado == 'Agresivo') and (objetivo_seleccionado == 'Crecimiento'):
        # Definir los umbrales y rangos para las métricas
        annual_returns_range = (0.25, 0.35)
        annual_volatility_threshold = 0.45
        daily_volatility_threshold = 0.030
        beta_threshold = 1
        market_cap_threshold = 20000000000
        daily_volume_threshold = 5000000

        # Filtrar las compañías que cumplan con las condiciones definidas
        aggressive_growth_companies = []
        for ticker in tickers:
            if (
                avg_anual_volatility[ticker] <= annual_volatility_threshold
                and avg_daily_volatility[ticker] <= daily_volatility_threshold
                and beta[ticker] >= beta_threshold
                and avg_anual_returns[ticker] >= annual_returns_range[0]
                and avg_anual_returns[ticker] <= annual_returns_range[1]
                and market_cap[ticker] >= market_cap_threshold
                and daily_volume[ticker] >= daily_volume_threshold
            ):
                aggressive_growth_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not aggressive_growth_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
        else:
            st.write(f'Compañías recomendadas para el perfil {perfil_seleccionado} con objetivo {objetivo_seleccionado}:')
            for ticker in aggressive_growth_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')


    elif (perfil_seleccionado == 'Especulativo') and (objetivo_seleccionado == 'Ingreso'):
        # Definir los umbrales y rangos para las métricas
        annual_returns_range = (0.33, 0.5)
        annual_volatility_threshold = 0.47
        daily_volatility_threshold = 0.050
        beta_threshold = 1.5
        market_cap_threshold = 2500000000
        daily_volume_threshold = 2500000

        # Filtrar las compañías que cumplan con las condiciones definidas
        speculative_income_companies = []
        for ticker in tickers:
            if (
                avg_daily_volatility[ticker] <= daily_volatility_threshold
                and avg_anual_volatility[ticker] <= annual_volatility_threshold
                and beta[ticker] <= beta_threshold
                and annual_returns_range[0] <= avg_anual_returns[ticker] <= annual_returns_range[1]
                and market_cap[ticker] >= market_cap_threshold
                and daily_volume[ticker] >= daily_volume_threshold
            ):
                speculative_income_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not speculative_income_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
        else:
            st.write(f'Compañías recomendadas para el perfil {perfil_seleccionado} con objetivo {objetivo_seleccionado}:')
            for ticker in speculative_income_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')

    elif (perfil_seleccionado == 'Especulativo') and (objetivo_seleccionado == 'Crecimiento'):
        # Definir los umbrales y rangos para las métricas
        annual_returns_range = (0.5, 1.4)
        annual_volatility_threshold = 0.65
        daily_volatility_threshold = 0.065
        beta_threshold = 1.4
        market_cap_threshold = 1000000000
        daily_volume_threshold = 3000000

        # Filtrar las compañías que cumplan con las condiciones definidas
        speculative_growth_companies = []
        for ticker in tickers:
            if (
                avg_anual_volatility[ticker] <= annual_volatility_threshold
                and avg_daily_volatility[ticker] <= daily_volatility_threshold
                and beta[ticker] >= beta_threshold
                and avg_anual_returns[ticker] >= annual_returns_range[0]
                and avg_anual_returns[ticker] <= annual_returns_range[1]
                and market_cap[ticker] >= market_cap_threshold
                and daily_volume[ticker] >= daily_volume_threshold
            ):
                speculative_growth_companies.append(ticker)

        # Mostrar las compañías recomendadas
        if not speculative_growth_companies:
            st.write('No se encontraron empresas que cumplan con los criterios definidos.')
        else:
            st.write(f'Compañías recomendadas para el perfil {perfil_seleccionado} con objetivo {objetivo_seleccionado}:')
            for ticker in speculative_growth_companies:
                st.write(f'{ticker} - Día recomendado para invertir: {dia_invertir[ticker]}')

elif opcion == 'KPIs':
    st.header('KPIs')
    
    # KPI 1: Ratio de éxito de las recomendaciones
    st.subheader('1. Ratio de éxito de las recomendaciones')
    st.write('Proporción de empresas que se recomendaron y que tuvieron un desempeño anual igual o superior al rango mínimo definido para esa combinación de perfil con objetivo, desde el día recomendado hasta la fecha actual.')
    st.write('Este KPI se puede calcular como la proporción de empresas recomendadas que han tenido un rendimiento anual igual o superior al rango mínimo definido, dividido entre el total de empresas recomendadas.')

    
    # KPI 2: Rentabilidad media de las recomendaciones
    st.subheader('2. Rentabilidad media de las recomendaciones')
    st.write('Este KPI nos indicaría cuánto hemos ganado en promedio con las recomendaciones que se han dado. Podríamos calcular la rentabilidad media como el promedio de los rendimientos obtenidos por todas las empresas recomendadas.')
    

    
    # KPI 3: Nivel de riesgo de las recomendaciones
    st.subheader('3. Nivel de riesgo de las recomendaciones')
    st.write('Este KPI nos indicaría el nivel de riesgo que hemos tomado en nuestras recomendaciones. Podríamos calcular el nivel de riesgo como la desviación estándar de los rendimientos obtenidos por todas las empresas recomendadas.')
