
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
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
----- Files que estan en path

"""


files=fn.f_files(path)

data_files=fn.data(files)


"""
----- Fechas de los csv

"""


data_dates=fn.f_dates()



"""
----- Tickers de los csv

"""


data_tickers=fn.f_tickers()

"""
----- Datos que importe de Yahoo finance que guarde en un csv para no estar corriendo Yfinance

"""

Data_down=fn.f_down_data().head()


"""
----- Datos de los filtros de las fechas que necesitabamos que eran por mes y no diarias

"""


Data_sort=fn.f_sortdates(Data_down,data_dates)


"""
----- Datos que continen los tickers en orden alfabético

"""

Precios_sort=fn.f_prices(data_files)

"""
----- Datos que continen los pesos

"""

Pesos_1=fn.f_pesos()

"""
----- Contiene el DataFrame de la invesrión pasiva

"""

pasiva= fn.f_df2(Precios_sort,Pesos_1,Data_sort,c_0,com)

"""
----- Contiene el DataFrame de los rendimientos y rendimientos acumulados de la invesrión pasiva

"""

rends = fn.rend(data_dates,pasiva).dropna()

"""
----- Contiene el dinero de los activos que eliminamos, de las acciones que no aparecían en todos los csv

"""


cash=((100-pasiva['Peso %'].values.sum())/100)*c_0


"""
----- Contiene la sumatoria de las comisiones de pasiva

"""


comisiones= (pasiva['Comisiones']*10).sum()

"""
----- Contiene la selección de tickers de la inversión pasiva

"""


tickers= pasiva.iloc[:,0]
tickers= tickers.to_list()

"""
----- Contiene una nueva selección de datos para trabajar el portafolio eficiente

"""


Data_activ=fn.f_down_data().iloc[0:251,:].set_index('Date')


"""
----- Contiene el Dataframe de la media y la volatilidad de los datos  

"""


ret_sum= fn.portafolio(Data_activ)


"""
----- Contiene la corralción de los datos y definimos la rf

"""

corr = Data_activ.corr()
rf = 0.085


"""
----- Contiene la matriz de covarianza de la volatilidad y los rendimientos esperados con la media

"""


## Construcción de parámetros
# 1. Sigma: matriz de varianza-covarianza Sigma = S.dot(corr).dot(S)

S= np.diag(ret_su.loc['Volatilidad'].values)
Sigma = S.dot(corr).dot(S)

# 2. Eind: rendimientos esperados activos individuales

Eind = ret_su.loc['Media'].values


"""
----- Contiene el número de activos, el w0 inicial y restricciones

"""

# Número de activos

N= len(Eind)

# Dato inicial

w0=np.ones(N) / N

# Cuotas de las variables

bnds = ((0,1),)*N

# Restricciones
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)

"""
----- Contiene el portafolio de mínima varianza

"""

minvar = minimize(fn.var(w, Sigma), w0, args=(Sigma,),bounds=bnds, constraints=cons)


"""
----- Contiene los pesos, el rendimiento, la volatilidad y el RS del poratfolio de mínima varianza

"""


# Pesos, rendimiento y riesgo del portafolio de mínima varianza

w_minvar = minvar.x
E_minvar =Eind.T.dot(w_minvar)
s_minvar = var(w_minvar, Sigma)**0.5
RS_minvar= (E_minvar - rf) / s_minvar


"""
----- Contiene los datos para encontrar el portafolio que máximiza el RS

"""
# Número de activos

N= len(Eind)

# Dato inicial

w0=np.ones(N) / N

# Cotas de las variables

bnds = ((0,1),)*N

# Restricciones
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)

"""
----- Contiene el portafolio de EMV:RS

"""


emv = minimize(fn.menos_RS(w,Eind, rf, Sigma), w0, args=(Eind,rf,Sigma),bounds=bnds, constraints=cons)


"""
----- Contiene pesos, rendimientos, volatilidades y el RS del portafolio EMV

"""

# Pesos, rendimiento y riesgo del portafolio EMV:MÁXIMO SR
w_emv = emv.x
E_emv= Eind.T.dot(w_emv)
s_emv = var(w_emv,Sigma)**0.5
RS_emv = (E_emv - rf)/s_emv

"""
----- Contiene la covarianza de los datos
"""

#Covarianza de los datos
cov_emv_minvar = w_emv.T.dot(Sigma).dot(w_minvar)


"""
----- Contiene la correlación de los datos
"""

# Correlacion de los datos

corr_emv_minvar = cov_emv_minvar /(s_emv*s_minvar)


"""
----- Contiene el vector de pesos w_p
"""

# Vector de w

w_p = np.linspace(0,1)

"""
----- Contiene tabla con media, volatilidad y RS
"""

frontera1=fn.frontera1(w_p,E_emv,E_minvar,s_emv,s_minvar,cov_emv_minvar,rf)

"""

-----  Dibujamos la LAC, combinando el portafolio EMV con el activo libre de riesgo, contiene volatilidad y media

"""

sp = np.linspace(0,0.2)

LAC1=fn.LAC1(sp,RS_emv,rf)

"""

-----  Combinación óptima de acuerdo a preferencias: Con los datos anteriores, y la caracterización de aversión al riesgo, se escoge la combinación óptima entre el portafolio EMV y el activo libre de riesgo

"""

g=8
w_opt =(E_emv-rf)/(g*s_emv**2)


optimo=w_opt *w_emv, w_opt + (1-w_emv), 1-w_opt


"""

-----  Contien los datos para elaborar el Dataframe del portafolio eficiente y el capital 

"""


prices_post1 = Data_activ.loc[Data_activ.index >= '2021-01-29',:]
lista = prices_post1.index.values

capital = (1000000 - cash)


portafolio_1=fn.portafolio1(tickers,prices_post1,Data_activ,pasiva,w_emv,capital,com)
portafolio_1


"""

-----  Al cash le resto las comisiones

"""

cash = cash - portfolio_1["Comisiones"].sum()

"""

-----  Seleccion de datos para la activa de acuerdo a las fechas establecidas del 2021-01-31 al 2022-07-29

"""

Data_activ1=f_down_data().iloc[251:].set_index('Date')

"""

-----  Seleccion de las fechas a trabajar para la activa y el cambio en los precios para el rebalanceo

"""


datesaño2=data_dates[12:]
cambio1= Data_activ1.pct_change()
cambio2= cambio1.dropna().reset_index()
cambio2=cambio2[cambio2.Date.isin (datesaño2)].reset_index(drop=True).set_index('Date')
cambio3=cambio2.reset_index(drop=True)

"""

-----  Contiene el Dataframe de la inversión activa

"""

inversion_activa=fn.activa(tickers,Data_activ1,portafolio_1,cambio2,cash,com)

"""

----- Contiene el Dataframe de los rendimientos y rendimientos acumulados de la inversión activa

"""


rends_activa=fn.rend_activa(datesaño2,inversion_activa)


"""

----- Contiene las operaciones historicas de la inversión activa

"""

hist_opera=fn.historico(datesaño2,inversion_activa,com)

"""

----- Contiene las medidas de desempeño de las inversiones activas y pasivas para su comparación

"""

medidasDesemp=fn.medidas(rends,rends_activa)







