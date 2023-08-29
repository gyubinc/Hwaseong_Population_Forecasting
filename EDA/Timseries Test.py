###########################

import pandas as pd
import numpy as np
import os

# plot 한글 안깨지게
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Users/jhmok/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


## 파일 불러오기
df = pd.read_excel("./data/Hwaseong_data.xlsx")


###########################


###########################

## 1차 차분
data = []
first_diff = [None]
data = df.iloc[:,2]
for i in range(len(data)-1) :
    first_diff.append(data[i+1] - data[i])
first_diff = pd.DataFrame({'first_diff' : first_diff}, index = df.index)
df = pd.concat([df, first_diff], axis = 1)

## 1차 로그차분
data = []
first_log_diff = [None]
data = df["총인구"]
for i in range(len(data)-1) :
    first_log_diff.append(np.log(data[i+1]) - np.log(data[i]))
first_log_diff = pd.DataFrame({'first_log_diff' : first_log_diff}, index = df.index)
df = pd.concat([df, first_log_diff], axis = 1)

## 2차 차분
second_diff = [None]
data = df["first_diff"]
for i in range(len(data)-1) :
    second_diff.append(data[i+1] - data[i])
second_diff = pd.DataFrame({"second_diff" : second_diff}, index = df.index)
df = pd.concat([df, second_diff], axis = 1)



total_pop = df["총인구"]
first_diff_pop = df["first_diff"].dropna()
first_log_diff_pop = df["first_log_diff"].dropna()
second_diff_pop = df["second_diff"].dropna()

total_pop_df = df.loc[:,["월별", "총인구"]]


###########################


###########################

## 그래프 확인
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(total_pop, color = "b")
plt.grid(True)
plt.title("총인구")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(first_diff_pop, color = "b")
plt.grid(True)
plt.title("1차 차분")
plt.show()

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(first_log_diff_pop, color = "b")
plt.grid(True)
plt.title("1차 로그 차분")
plt.show()

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(second_diff_pop, color = "b")
plt.grid(True)
plt.title("2차 차분")
plt.show()

###########################

###########################
## ACF, PACF plot (인구수)
# 데이터끼리 의존관계가 있음. 11월 시차까지
fig, ax = plt.subplots(figsize =(6, 4))
plot_acf(total_pop, color = "b")
plt.grid(True)
plt.title("총인구 ACF")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plot_acf(first_diff_pop, color = "b")
plt.grid(True)
plt.title("1차 차분 ACF")
plt.show()




## ADF 검정 (lag = 4, regression ct)
def adf_test(timeseries):
    print("ADF test:")
    dftest = adfuller(timeseries, regression = "ct", maxlag = 4, autolag = None)
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)



# 총인구에 대한 ADF test
adf_test(total_pop)
# 1차 차분에 대한 ADF test
adf_test(first_diff_pop)
# 1차 로그 차분에 대한 ADF test
adf_test(first_log_diff_pop)
# 2차 차분에 대한 ADF test (유의)
adf_test(second_diff_pop)




## KPSS test (lag = 4, regression ct)
def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="ct", nlags=4)
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)
    
    
# 총인구에 대한 KPSS test
kpss_test(total_pop)
# 1차 차분에 대한 KPSS test
kpss_test(first_diff_pop)
# 1차 로그 차분에 대한 KPSS test
kpss_test(first_log_diff_pop)
# 2차 차분에 대한 KPSS test (유의)
kpss_test(second_diff_pop)



###########################

## AR 돌려보기
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(total_pop_df["총인구"], order = (4,1,0))
fit_model = model.fit()
print(fit_model.summary())


forecast_steps = 32
forecast = fit_model.forecast(steps=forecast_steps)

print("Forecasted values:", forecast)

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(total_pop_df.index, total_pop_df["총인구"], color = "b")
plt.plot(forecast.index, forecast.values, color = "r")
plt.title("총인구")
plt.show()


##
model = ARIMA(total_pop_df["총인구"], order = (4,2,0))
fit_model = model.fit()
print(fit_model.summary())


forecast_steps = 32
forecast = fit_model.forecast(steps=forecast_steps)

print("Forecasted values:", forecast)


fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(total_pop_df.index, total_pop_df["총인구"], color = "b")
plt.plot(forecast.index, forecast.values, color = "r")
plt.title("총인구(2차 차분)")
plt.show()



##
model = ARIMA(np.log(total_pop_df["총인구"]), order = (4,1,0))
fit_model = model.fit()
print(fit_model.summary())


forecast_steps = 32
forecast = fit_model.forecast(steps=forecast_steps)

print("Forecasted values:", forecast)


fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(total_pop_df.index, total_pop_df["총인구"], color = "b")
plt.plot(forecast.index, np.exp(forecast.values), color = "r")
plt.title("총인구 (1차 로그 차분)")
plt.show()









