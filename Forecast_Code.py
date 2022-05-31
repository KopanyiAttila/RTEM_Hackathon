# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:03:32 2022

@author: Lenovo
"""

!pip install onboard.client 

import pandas as pd 

from onboard.client import RtemClient import matplotlib.pyplot as plt 

import seaborn as sns plt.style.use('ggplot')  plt.rcParams["figure.figsize"] = (20,9)  

api_key = ‘input your key' 

client = RtemClient(api_key=api_key) 

df_temp_raw = pd.read_csv("/kaggle/input/onboard-api-intro/fcu/fcu_discharge_air_temperature_20171001_20181001.csv", parse_dates=["timestamp"]) 

df_cmd_raw = pd.read_csv("/kaggle/input/onboard-api-intro/fcu/fcu_command_20171001_20181001.csv", parse_dates=["timestamp"]) 

sorted_cols = ["timestamp"] + sorted(df_temp_raw.columns[1:]) 

df_temp = df_temp_raw[sorted_cols] 

df_temp.columns 

df_temp = df_temp.resample("1H", on="timestamp").mean() 

cols426 = df_temp.columns[df_temp.columns.str.contains("426")] 

bdg426 = df_temp[cols426] 

import matplotlib.pyplot as plt 

import seaborn as sns 

plt.style.use('ggplot') 

plt.rcParams["figure.figsize"] = (20,9) 

sns.lineplot(data=bdg426) 

bdgs = pd.DataFrame(client.get_all_buildings()) 

bdgs[bdgs.id == 426] 

points = ["426_26941_283206"] 

sel_points = df_temp[points].reset_index() 

sel_points 

df1 = sel_points.melt(id_vars=["timestamp"]) 

df2 = df_temp_raw[["timestamp"] + points].reset_index(drop=True).resample("24H", on="timestamp").mean().reset_index() 

df2 = df2.melt(id_vars=["timestamp"]) 

plt.rcParams["figure.figsize"] = (16,4) 

fig5 = sns.lineplot(data=df2, x="timestamp", y="value", hue="variable") 

data = df2 

data= data.bfill() 

ds = df2['value'] 

ds1 = ds.dropna() 

data.index = pd.to_datetime(data['timestamp']) 

data.drop(columns='timestamp',inplace=True) 

data.head() 

!pip install pandas-datareader 

import pandas_datareader.data as web 

import datetime 

data.isna().sum() 

data.plot() 

from statsmodels.tsa.statespace.sarimax import SARIMAX 

from statsmodels.tsa.arima.model import ARIMA 

h = data.dropna()['value'] 

ts_train = h['2017-12-02':'2018-12-01'] 

ts_test = h['2018-05-01':] 

from statsmodels.tsa.seasonal import seasonal_decompose 

result = seasonal_decompose(h['2017-12-02':'2018-12-01'], model='additive',extrapolate_trend='freq') 

result.plot() 

plt.show() 

from statsmodels.tsa.stattools import adfuller 

import numpy as np 

def check_stationarity(h): 

    dftest = adfuller(h) 

    adf = dftest[0] 

    pvalue = dftest[1] 

    critical_value = dftest[4]['5%'] 

    if (pvalue < 0.05) and (adf < critical_value): 

        print("The series is stationary") 

    else: 

        print("The series is NOT stationary") 

seasonal = result.seasonal 

check_stationarity(seasonal) 

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 

plot_pacf(seasonal, lags =12) 

plt.show() 

plot_acf(seasonal, lags =12) 

plt.show() 

from statsmodels.tsa.statespace.sarimax import SARIMAX 

model = SARIMAX(ts_train, order=(5,0,1)) 

model_fit = model.fit() 

n_test = ts_test.shape[0] 

ts_pred = model_fit.forecast(steps=n_test) 

from statsmodels.tools.eval_measures import rmse 

nrmse = rmse(ts_pred, ts_test)/(np.max(ts_test)-np.min(ts_test)) 

nrmse 

model_seasonal = SARIMAX(ts_train, order=(5,0,1), seasonal_order=(12,0,8,12)) 

model_fit_seasonal = model_seasonal.fit() 

ts_pred_seasonal = model_fit_seasonal.forecast(steps=n_test) 

nrmse_seasonal = rmse(ts_pred_seasonal, ts_test)/(np.max(ts_test)-np.min(ts_test)) 

nrmse_seasonal 

N = 1 

ts_pred = model_fit.forecast(steps=n_test+N) 

ts_pred_seasonal = model_fit_seasonal.forecast(steps=n_test+N) 

plt.figure(figsize=(20,5)) 

plt.plot(ts_pred_seasonal, label='Prediction') 

plt.plot(h['2017-12-02':'2018-12-01'], label='Actual data FCU temperature (Building 426)') 

plt.title('Multi-step Forecasting (manual parameters)') 

plt.legend() 

plt.grid() 

plt.xticks(rotation=90) 

plt.show() 