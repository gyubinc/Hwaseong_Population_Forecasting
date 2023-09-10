
########################### 기본 설정 ###########################

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


########################### 차분 ###########################

total_pop = df["총인구"]
first_diff_pop = df["총인구"].diff()
first_log_diff_pop = np.log(df["총인구"]).diff()
second_diff_pop = df["총인구"].diff().diff()


df = pd.concat([df, second_diff_pop], axis = 1)
df = df.iloc[:,[0, -1]]



########################### 정상성 만족 ########################### 


###### 그래프 확인 ######

# 총인구 그래프 (정상성을 띄지 않는다)
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.월별, total_pop, color = "b")
plt.grid(True)
plt.title("총인구")
plt.show()

# 1차 차분 그래프 (정상성을 띄지 않는다)
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.월별, first_diff_pop, color = "b")
plt.grid(True)
plt.title("1차 차분")
plt.show()

# 1차 로그 차분 그래프 (정상성을 띄지 않는다)
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.월별, first_log_diff_pop, color = "b")
plt.grid(True)
plt.title("1차 로그 차분")
plt.show()

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.월별, second_diff_pop, color = "b")
plt.grid(True)
plt.title("2차 차분")
plt.show()


###### ADF, KPSS test ###### 

# ADF test (lag = 4, regression = "ct") 

def adf_test(timeseries, pvalue = 0.05):
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
    if dfoutput[1] < pvalue :
        return "정상 시계열"
    else :
        return "비정상 시계열"

# 총인구에 대한 ADF test
adf_test(total_pop)
# 1차 차분에 대한 ADF test
adf_test(first_diff_pop[1:,])
# 1차 로그 차분에 대한 ADF test
adf_test(first_log_diff_pop[1:,])
# 2차 차분에 대한 ADF test (유의)
adf_test(second_diff_pop[2:,])




# KPSS test (lag = 4, regression = "ct")
def kpss_test(timeseries, pvalue = 0.05):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="ct", nlags=4)
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)
    if kpss_output[1] > pvalue :
        return "정상 시계열"
    else :
        return "비정상 시계열"
    
# 총인구에 대한 KPSS test
kpss_test(total_pop)
# 1차 차분에 대한 KPSS test
kpss_test(first_diff_pop[1:,])
# 1차 로그 차분에 대한 KPSS test
kpss_test(first_log_diff_pop[1:,])
# 2차 차분에 대한 KPSS test (유의)
kpss_test(second_diff_pop[2:,])






###### ACF, PACF plot ###### 

# ACF가 서서히 감소하는 패턴 : AR 모델
# ACF가 급격히 감소하는 패턴 : MA 모델

# AR 쓴 이유 : baseline이여서
# lag = 0 인 이유 : ACF, PACF 기준 



## ACF, PACF plot (인구수)
# 데이터끼리 의존관계가 있음. 11월 시차까지
fig, ax = plt.subplots(figsize =(6, 4))
plot_acf(total_pop, color = "b")
plt.grid(True)
plt.title("총인구 ACF")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plot_acf(second_diff_pop[2:,], color = "b")
plt.grid(True)
plt.title("2차 차분 ACF")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plot_pacf(second_diff_pop[2:,], color = "b")
plt.grid(True)
plt.title("2차 차분 PACF")
plt.show()





########################### AR 모델 ###########################

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# AR 모델 적합
lag_order = 4 # AR 모델의 차수 설정
model = AutoReg(second_diff_pop[2:,], lags=lag_order)
results = model.fit()


# 모델로부터 예측값 얻기
predictions = results.predict(start=lag_order, end=110 - 1)

# 실제값
actual_values = second_diff_pop[(2+lag_order):,]

# RMSE 계산
rmse = np.sqrt(mean_squared_error(actual_values, predictions))
r2_score(actual_values, predictions)
print("RMSE:", rmse)

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(actual_values[2:,], color = "b")
plt.plot(predictions, color = "r")
plt.show()





# AR 모델 적합
lag_order = 2 # AR 모델의 차수 설정
model = AutoReg(total_pop, lags=lag_order)
results = model.fit()


# 모델로부터 예측값 얻기
predictions = results.predict(start=lag_order, end=110 - 1)

# 실제값
actual_values = total_pop[lag_order:]

# RMSE 계산
rmse = np.sqrt(mean_squared_error(actual_values[2:], predictions))
r2_score(actual_values[2:,], predictions)
print("RMSE:", rmse)

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(actual_values[2:,], color = "b")
plt.plot(predictions, color = "r")
plt.show()





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




model = ARIMA(second_diff_pop[2:,], order = (3,0,0))
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






