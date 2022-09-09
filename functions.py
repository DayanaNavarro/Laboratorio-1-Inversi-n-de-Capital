 """
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# --------------- TRATAMIENTO DE LOS DATOS -------------------

import numpy as np
import pandas as pd
import numpy as np
import time
from os.path import isfile, join
from os import listdir, path
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import glob
import pandas_datareader.data as web
from scipy.optimize import minimize

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

"""
---------- Obtención de los CSV de los archivos ---------------

"""

def f_files(path):
 
    files = glob.glob(path + "/*.csv")
    
    return files

"""
---------- Dataframe de los files, modificación de tickers---------------

"""

def data(files):

    data_frame = {}

    for i in files:
        data = pd.read_csv(i, skiprows=2, header=None)
        data.columns = list(data.iloc[0, :])
      
        data = data.loc[:, pd.notnull(data.columns)]
        data = data.iloc[1:-1].reset_index(drop=True, inplace=False)
    
        data['Precio'] = [i.replace(',', '') for i in data['Precio']]
        data['Ticker'] = [i.replace('*', '') for i in data['Ticker']]
       
        data = data.astype({'Ticker': str, 'Nombre': str, 'Peso (%)': float, 'Precio': float})

        data['Peso(%)'] = data['Peso (%)'] / 100
        data_frame[i] = data

    return data

"""
---------- Fechas de los archivos CSV---------------

"""

def f_dates():
    dates = ['2020-01-31','2020-02-28','2020-03-31','2020-04-30','2020-05-29','2020-06-30',
       '2020-07-31','2020-08-31','2020-09-30','2020-10-30','2020-11-30','2020-12-31',
       '2021-01-29','2021-02-26','2021-03-31','2021-04-30','2021-05-31','2021-06-30',
       '2021-07-30','2021-08-31','2021-09-30','2021-10-26','2021-11-30','2021-12-31',
       '2022-01-26', '2022-02-28','2022-03-31','2022-04-29','2022-05-31',
       '2022-06-30','2022-07-29']
    return dates

"""
----------Tickers de los archivos CSV---------------

"""

def f_tickers():
    tickers=['AMXL.MX','WALMEX.MX','GFNORTEO.MX','GMEXICOB.MX','FEMSAUBD.MX','CEMEXCPO.MX',
       'GAPB.MX','TLEVISACPO.MX','BIMBOA.MX','ASURB.MX','ELEKTRA.MX','GFINBURO.MX', 'KOFUBL.MX',
       'AC.MX','GRUMAB.MX','ORBIA.MX','KIMBERA.MX','ALFAA.MX','OMAB.MX','BBAJIOO.MX',
       'PINFRA.MX','GCARSOA1.MX','PE&OLES.MX','CUERVO.MX','ALSEA.MX',
       'LIVEPOLC-1.MX','BOLSAA.MX','MEGACPO.MX','LABB.MX','MXN=X']
    return tickers


"""
---------- Descarga de datos de yahoo finance y guardado en un csv---------------

"""

def f_down_data():
    
    DataPos=pd.read_csv('Prices.csv')
    
    return DataPos 

"""
---------- Filtramos las fechas diarias con las fechas de los csv obteniendo así los precios de las fechas que necesitamos-------

"""


def f_sortdates(Data_down,data_dates):
    
    precios = pd.DataFrame(Data_down)
    precios=precios[precios.Date.isin(data_dates)].reset_index(drop=True)
    
    return precios


"""
---------- Se orden los tickers alfabéticamente ---------------

"""


def f_prices(data_files):
    
    by_Ticker = data_files.iloc[0:36].sort_values('Ticker')
    
    return by_Ticker

"""
---------- Pesos ---------------

"""


def f_pesos():
    
    pesos1=[1.77, 1.51,0.86,13.70,2.75,0.83,1.87,0.72,4.30,0.67,3.02,11.85,3.29,1.16,1.76,10.64,6.03,1.68,
                   2.06,2.26,0.62,0.74,0.79,0.27,1.85,1.67,0.91,1.68,3.96,10.39]
    return pesos1

"""
---------- Dataframe de la inversión pasiva --------------------------

DataFrame con los datos de ticker, precio, peso, capital necesario a invertir,
titulos a comprar,la postura y las comisiones que nos cobrarían.
Además de la eliminación de los Tickers que no estaban en todos los archivos.

"""


def f_df2(Precios_sort,Pesos_1,Data_sort,c_0,com):


    df2 = pd.DataFrame()

    df2['Ticker']= Precios_sort['Ticker']
    
    df2.drop(df2[(df2.Ticker != 'AC') & (df2.Ticker != 'ALFAA') & (df2.Ticker != 'ALSEA') &
            (df2.Ticker !='AMXL') & (df2.Ticker != 'ASURB') & (df2.Ticker != 'BBAJIOO') &
             (df2.Ticker != 'BIMBOA')& (df2.Ticker != 'BOLSAA') & (df2.Ticker != 'CEMEXCPO') &
             (df2.Ticker != 'CUERVO')&(df2.Ticker != 'ELEKTRA') & (df2.Ticker != 'FEMSAUBD') & 
             (df2.Ticker != 'GAPB')& (df2.Ticker!= 'GCARSOA1') & (df2.Ticker != 'GFINBURO') & 
             (df2.Ticker != 'GFNORTEO')&(df2.Ticker != 'GMEXICOB') & (df2.Ticker != 'GRUMAB') & 
             (df2.Ticker != 'KIMBERA')&(df2.Ticker != 'KOFUBL') & (df2.Ticker != 'LABB') & 
             (df2.Ticker != 'LIVEPOLC.1') & (df2.Ticker != 'MEGACPO') & (df2.Ticker != 'MXN') & 
             (df2.Ticker != 'OMAB')& (df2.Ticker != 'ORBIA') & (df2.Ticker != 'PE&OLES') & 
             (df2.Ticker != 'PINFRA') & (df2.Ticker != 'TLEVISACPO') & (df2.Ticker != 'WALMEX') ].index, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    
    df2['Peso %']= Pesos_1
    
    df2['Ticker'] = df2['Ticker'] + '.MX'
    
    df2.iloc[21,0]='LIVEPOLC-1.MX'
    
    df2.iloc[23,0]='MXN=X'
    
    df2['Precio']=(np.array([Data_sort.iloc[0, Data_sort.columns.to_list().index(i)] for i in df2['Ticker']]))
    
    df2['Titulos'] = np.floor((c_0 * df2['Peso %']) / (df2['Precio'] + (df2['Precio'] * com))/100)
    
    df2['Capital'] = np.round(df2['Titulos'] * (df2['Precio'] + (df2['Precio'] * com)), 2)
    
    df2['Postura'] = np.round(df2['Peso %'] * df2['Capital'], 2)
    
    df2['Comisiones'] = np.round(df2['Precio'] * com * df2['Titulos'], 2)


    return df2

"""
---------- Dataframe de la inversión pasiva rendimientos y rendimientos acumulados --------------------------

"""

def rend(data_dates,pasiva):
    
    df3 = pd.DataFrame()
    df3['Date']= data_dates
    
    df3['Capital'] =pasiva['Capital']
    
    
    df3['Rend'] = 0

    df3['Rend Acum'] = 0


    #Rentabilidad simple = (Valor final de inversión – Valor inicial de la inversión) / Valor inicial de la inversión

    #Rentabilidad acumulada = suma de rendimientos individuales

    for i in range(1, len(df3)):
    
        df3.loc[i, "Rend"] = (df3.loc[i, 'Capital'] - df3.loc[i - 1, 'Capital']) / df3.loc[i - 1, 'Capital']
    
        df3.loc[i, "Rend Acum"] = df3.loc[i, 'Rend'] + df3.loc[i - 1, 'Rend Acum']


    return df3


"""
 ---------Función que regresa el portafolio eficiente, obteniendo el radio de Sharpe------------
"""


"""
Función de la media y volatilidad de los precios 
"""


def portafolio(Data_activ):
    
    annual_ret_summ = pd.DataFrame(columns=Data_activ.columns)
    annual_ret_summ.loc['Media'] = Data_activ.mean()
    annual_ret_summ.loc['Volatilidad'] = Data_activ.var()
    
    return annual_ret_summ

"""
Función de la minima varianza
"""


# Función objetivo
def var(w, Sigma):
    return w.T.dot(Sigma).dot(w)

"""
Función del Radio de Sharpe
"""

# Función objetivo
def menos_RS(w,Eind, rf, Sigma):
    E_port = Eind.T.dot(w)
    s_port = var(w,Sigma)**0.5
    RS =(E_port- rf)/s_port
    return -RS

"""
Función de la frontera

"""

def frontera1(w_p,E_emv,E_minvar,s_emv,s_minvar,cov_emv_minvar,rf):
    frontera = pd.DataFrame(data={'Media' : w_p*E_emv + (1-w_p)*E_minvar,
                                 'Vol': ((w_p*s_emv)*2 +((1-w_p)*s_minvar)**2 + 2 * w_p * (1-w_p)*cov_emv_minvar)*0.5})

    frontera['RS']=(frontera['Media'] - rf)/frontera['Vol']
    
    return frontera

"""
Gráfica de dispersión de puntos coloreando de acuerdo a SR, y portafolio EMV

"""


def grafica1(frontera1,ret_sum,s_minvar,E_minvar,s_emv, E_emv):

    plt.figure(figsize=(10,6))
    # Frontera
    plt.scatter(frontera1['Vol'], frontera1['Media'], c=frontera1['RS'], cmap = 'RdYlBu', label = 'Frontera de min var')
    plt.colorbar()

    # Activos ind
    for activo in list(ret_sum.columns):
        plt.plot(ret_sum.loc['Volatilidad', activo],
                 ret_sum.loc['Media', activo],
                 'o',
                 ms=5,
                label = activo)

    # Port. óptimos
    plt.plot(s_minvar, E_minvar, '*g', ms=10, label='Portafolio de min var')
    plt.plot(s_emv, E_emv, '*r', ms=10, label='Portafolio eficiente en media var')
    plt.xlabel('Volatilidad $\sigma$')
    plt.ylabel('Rendimiento Esperado $E[r]$')
    plt.grid()
    plt.legend(loc='best')


"""

LAC, combinación del portafolio EMV con el activo libre de riesgo

"""
   

def LAC1(sp,RS_emv,rf):

    LAC= pd.DataFrame(data={'Vol': sp,
                           'Media': RS_emv*sp+rf})
    return LAC.head()
"""

Gráfica LAC La gráfica de LAC describe las posibles selecciones de riesgo-rendimiento entre un activo libre de riesgo y un activo riesgoso.

"""



def grafica2(frontera1,s_emv,E_emv,LAC1):
    plt.figure(figsize=(6,4))
    plt.scatter(frontera1['Vol'], frontera1['Media'], c=frontera1['RS'], cmap='RdYlBu', label='PORTAFOLIOS')
    plt.plot(s_emv, E_emv,'*y', ms=10, label='PORTAFOLIO EMV')
    plt.plot(LAC1['Vol'], LAC1['Media'], label='LAC')
    plt.grid()
    plt.xlabel('VOLATILIDAD')
    plt.ylabel('MEDIA')
    plt.legend(loc='best')
    plt.colorbar()
    
    
"""

Portafolio eficiente, con pesos de RS y precio del 2021-01-29, con datos del 2020 al 2021

"""
  
def portafolio1(tickers,prices_post1,Data_activ,pasiva,w_emv,capital,com):

    portfolio_1 = pd.DataFrame()
    portfolio_1["Ticker"] = tickers
    portfolio_1['Precios'] = (np.array([prices_post1.iloc[0, Data_activ.columns.to_list().index(i)] for i in pasiva['Ticker']]))
    portfolio_1["Peso"] = w_emv
    portfolio_1['Postura'] = np.round(capital * portfolio_1["Peso"], 2)
    portfolio_1['Titulos'] = np.floor((portfolio_1["Postura"] / portfolio_1["Precios"]))
    portfolio_1['Comisiones'] =np.round(portfolio_1['Precios'] * com * portfolio_1['Titulos'], 2)
    portfolio_1 = portfolio_1.set_index("Ticker")

    return portfolio_1


"""

Dataframe Inversión Activa. Es un dataframe que generá lo siguiente

**VENTA**

- Los **Titulos_antV** son lso titulos antes de la venta, los titulos anteriores
- Los **Titulos_V** son con un for, en mi len(cambio2) donde mi cambio2 es el cambio en los precios, entonces lo que ese for pretende hacer es que en cada cambio me diga si subio o bajo de acuerdo a los rebalanceos de 5% y los guarda en una lista. De ahi con un if, si el cambio en tickers esta down(bajo) vende, entonces se lo resta a los **Titulos_antV**.
- La **Venta** entonces es la diferencia de esos titulos por el precio
- Las **Comisiones venta** son el resultado de la multiplicación de la Venta por la comision definida al inicio.

**COMPRA**

- Los **Titulos_antC** de acuerdo a mi lógica serían los titulos **Titulos_V**, ya que esos titulos que vendí serían los que estarían disponibles para la compra.
- Los **Titulos_C** son con el for anterior que igual recorre len(cambio2) que es el cambio en los precios, entonces lo que ese for pretende hacer es que en cada cambio me diga si subio o bajo de acuerdo a los rebalanceos de 5% y los guarda en una lista. De ahi con un if, si el cambio en tickers esta  up(arriba) compra, entonces se lo suma a los **Titulos_C**.
- La **Compras** entonces es la diferencia de esos titulos por el precio
- Las **Comisiones compra** son el resultado de la multiplicación de las Compras por la comision definida al inicio.

**CAPITAL**

- De acuerdo a mi lógica, lo fui guardando, primero como *dinero_venta* que eso es: mi cash ya definido más la **Venta** menos la suma de **Comisiones venta**  y ese *dinero_venta*  es lo que gane de vender.
- Y ya que tengo ese dinero de la venta ahora si puedo comprar que sería el *dinero_compra*, que es igual al dinero_venta pero con **Compras** más las **Comisiones compra**
- Y por ultimo **Capital** que quedaría después de esas transacciones sería mi cash más esos **Titulos_C** por el precio

"""


def activa(tickers,Data_activ1,portafolio_1,cambio2,cash,com):
    
    new_portfolio = pd.DataFrame()
    new_portfolio["Ticker"] = tickers
    new_portfolio = new_portfolio.set_index("Ticker")
    new_portfolio["Precio"] = Data_activ1.iloc[251,:].to_list()

    new_portfolio["Titulos_antV"]=portafolio_1.loc[:, "Titulos"].to_list()

    new_portfolio["Titulos_V"]=0

    for i in range(len(cambio2)):
        cambio = pd.DataFrame(cambio2.iloc[i,:])
        cambio.columns=["Rebalanceo"]
        pdown = cambio[cambio.Rebalanceo <.05 ]
        down = list(pdown.index.values)
        pup = cambio[cambio.Rebalanceo > .05 ]
        up = list(pup.index.values)

    #VENTA

    for ticker in tickers:

            if ticker in down:

                new_portfolio["Titulos_V"]=new_portfolio.loc[tickers,"Titulos_antV"]*(1-0.025)

    new_portfolio["Ventas"] = np.round((new_portfolio["Titulos_antV"] - new_portfolio["Titulos_V"]) * new_portfolio["Precio"], 2)
    new_portfolio["Comisiones venta"] = new_portfolio["Ventas"] * com 


    dinero_venta=cash+new_portfolio["Ventas"].sum()-new_portfolio["Comisiones venta"].sum()


    #COMPRA


    new_portfolio["Titulos_antC"] =new_portfolio["Titulos_V"]

    new_portfolio["Titulos_C"]=0

    for ticker in tickers:

            if ticker in up:

                new_portfolio["Titulos_C"]=new_portfolio.loc[tickers,"Titulos_antC"]*(1+0.025)

    new_portfolio["Compras"] = np.round((new_portfolio["Titulos_antC"] - new_portfolio["Titulos_C"]) * new_portfolio["Precio"], 2)
    new_portfolio["Comisiones compra"] = new_portfolio["Compras"] * com 

    dinero_compra= dinero_venta - new_portfolio["Compras"].sum() + new_portfolio["Comisiones compra"].sum()

    new_portfolio["Capital"]= cash + new_portfolio["Titulos_C"]*new_portfolio["Precio"]


    return new_portfolio

"""

Rendimientos y Rendimientos acumulados de la inversión activa

 - Rentabilidad simple = (Valor final de inversión – Valor inicial de la inversión) / Valor inicial de la inversión

 - Rentabilidad acumulada =  suma de rendimientos individuales

"""

def rend_activa(datesaño2,inversion_activa):

    df5 = pd.DataFrame()

    df5['Date']= datesaño2

    df5["Capital"]=inversion_activa["Capital"].reset_index(drop=True)

    df5["Rend"]=0

    df5["Rend Acum"]=0


    for i in range(1, len(df5)):

            df5.loc[i, "Rend"] = (df5.loc[i, 'Capital'] - df5.loc[i - 1, 'Capital']) / df5.loc[i - 1, 'Capital']

            df5.loc[i, "Rend Acum"] = df5.loc[i, 'Rend'] + df5.loc[i - 1, 'Rend Acum']


    return df5


"""

Histoticos Operaciones de la inversión activa

> Para esta tabla de histórico de operaciones, se realizo lo siguiente, de acuerdo a las requisiciones de laboritario debía de tener titulos totales, comisiciones totales y acumuladas, lo que hice fue que con los datos del dataframe inversión activa arriba hecho, multilique esos **Titulos_V** y **Titulos_C** por el precio y sume esos resultados para obtener **Titulos Totales**.

> Y para las comisiones igual multiplique las **Ventas** y **Compras** por la com que ya teniamos definida arriba y eso lo sume para obtener **Comisiones**.

> Y por último **Comisiones acumuladas**, fue el resultado de aplicarle a **Comisiones** un **cumsum()** que lo que hace es retornar las sumas acumuladas de los elementos.

"""


def historico(datesaño2,inversion_activa,com):
    
    df6 = pd.DataFrame()

    df6['Date']= datesaño2

    df6["Titulos Venta"]= (inversion_activa["Titulos_V"]*inversion_activa["Precio"]).reset_index(drop=True)

    df6["Titulos Compra"]= (inversion_activa["Titulos_C"]*inversion_activa["Precio"]).reset_index(drop=True)

    df6["Titulos Totales"]= df6["Titulos Venta"]+df6["Titulos Compra"]

    df6["Titulos Venta"]= (inversion_activa["Titulos_V"]*inversion_activa["Precio"]).reset_index(drop=True)

    df6["Comision Venta"]= (inversion_activa["Ventas"] * com ).reset_index(drop=True)

    df6["Comision Compra"]= (inversion_activa["Compras"] * com).reset_index(drop=True)*-1

    df6["Comisiones"]= df6["Comision Venta"]+df6["Comision Compra"]

    df6["Comisiones acumuladas"]=df6["Comisiones"].cumsum()





    return df6


"""

Medidas de desempeño con los datos anteriores ya obtenidos con la pasiva y la activa

"""

def medidas(rends,rends_activa):  
    
    pasiva2=rends["Rend"].mean()/100
    pasiva3=rends["Rend Acum"].mean()/100
    activa2=rends_activa["Rend"].mean()
    activa3=rends_activa["Rend Acum"].mean()

    datos = {
        'Medidas' : ['rend_m', 'rend_c'],
        'Descripcion': ['Rendimiento Promedio Mensual', 'Rendimiento mensual acumulado'],
        'Inv_pasiva': [pasiva2, pasiva3],
        'Inv_activa': [activa2, activa3]
    }

    df7 = pd.DataFrame(datos)
    return df7





