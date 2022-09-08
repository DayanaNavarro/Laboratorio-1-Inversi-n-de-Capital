
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

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
-------------En este script dejas todas las funciones desarrolladas para visualizar datos, ya sean gráficas, tablas, textos impresos y/o híbridos entre estos elementos.
"""

"""
 DataFrame de las datos de yahoo finance y guardado en un csv

"""

def f_down_data():
    
    DataPos=pd.read_csv('Prices.csv')
    
    return DataPos 

"""
 Dataframe de la inversión pasiva 

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


 Dataframe de la inversión pasiva rendimientos y rendimientos acumulados 

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

DataFrame Portafolio eficiente, con pesos de RS y precio del 2021-01-29, con datos del 2020 al 2021

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

DataFrame Rendimientos y Rendimientos acumulados de la inversión activa

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

DataFrame Histoticos Operaciones de la inversión activa

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

DataFrame Medidas de desempeño con los datos anteriores ya obtenidos con la pasiva y la activa

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






