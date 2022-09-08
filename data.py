
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
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
import functions as fn
import yfinance as yf
import glob
import pandas_datareader.data as web
from scipy.optimize import minimize

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)


"""
Variable que guarda las fechas

"""
data_dates=fn.f_dates()


"""
Variable que guarda los tickers

"""
data_tickers=fn.f_tickers()


"""
Variable que guarda los datos descargados de yahoo finance

"""
Data_down=fn.f_down_data().head()

"""
Variable que guarda los datos filtrados de diarios a mensuales

"""
Data_sort=fn.f_sortdates(Data_down,data_dates)

"""
Variable que ordena los datos alfabeticamente

"""

Precios_sort=fn.f_prices(data_files)

"""
Variable que contiene los pesos

"""

Pesos_1=fn.f_pesos()



"""
Variable que contiene los tickers para el portafolio eficiente

"""

tickers= pasiva.iloc[:,0]
tickers= tickers.to_list()


"""
Variable que contiene los datos descargados de yahoo finance

"""

Data_down=fn.f_down_data()


"""
Variable con la cantidad de datos a trabajar, que serían de 31-01-2022 al 31-01-2021

"""

Data_activ=fn.f_down_data().iloc[0:251,:].set_index('Date')


"""
Variable de pesos, rendimientos, volatilidad y RS de el portafolio mínima varianza

"""


minvar = minimize(fn.var(w, Sigma), w0, args=(Sigma,),bounds=bnds, constraints=cons)

w_minvar = minvar.x
E_minvar =Eind.T.dot(w_minvar)
s_minvar = var(w_minvar, Sigma)**0.5
RS_minvar= (E_minvar - rf) / s_minvar


"""
Variable de pesos, rendimientos, volatilidad y RS de el portafolio EMV:RS

"""

emv = minimize(fn.menos_RS(w,Eind, rf, Sigma), w0, args=(Eind,rf,Sigma),bounds=bnds, constraints=cons)

w_emv = emv.x
E_emv= Eind.T.dot(w_emv)
s_emv = var(w_emv,Sigma)**0.5
RS_emv = (E_emv - rf)/s_emv

"""
Variable que guarda la covarianza de los datos

"""


cov_emv_minvar = w_emv.T.dot(Sigma).dot(w_minvar)

"""
Variable que guarda la correlación de los datos

"""

corr_emv_minvar = cov_emv_minvar /(s_emv*s_minvar)

"""
Variable que contiene la combinación óptima entre el portafolio EMV:RS y el activo libre de riesgo

"""

optimo=w_opt *w_emv, w_opt + (1-w_emv), 1-w_opt


"""
Variable que contiene los precios para el portafolio eficiente y el capital

"""

prices_post1 = Data_activ.loc[Data_activ.index >= '2021-01-29',:]
lista = prices_post1.index.values


capital = (1000000 - cash)

"""
Variable que contiene los datos de los precios para el estudio de la pasiva

"""

Data_activ1=f_down_data().iloc[251:].set_index('Date')


"""
Variable que contiene las fechas para el tiempo de estudio de la activa

Variable de los cambios en los precios para rebalanceo

"""


datesaño2=data_dates[12:]
cambio1= Data_activ1.pct_change()
cambio2= cambio1.dropna().reset_index()
cambio2=cambio2[cambio2.Date.isin (datesaño2)].reset_index(drop=True).set_index('Date')
cambio3=cambio2.reset_index(drop=True)








