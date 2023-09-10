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

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.eval_measures import rmse


## 파일 불러오기
df = pd.read_excel("./data/찐 최종 데이터.xlsx")


# Initialize an empty DataFrame to store the rolling predictions
rolling_predictions_corrected = pd.DataFrame()

# Initialize the starting month of the training set
start_month = pd.Timestamp('2014-03-01')

# Loop to generate rolling predictions
while True:
    # Define the training and validation sets based on the current start month
    train_df = df[df['월별'] <= start_month + pd.DateOffset(months=59)]  # 59 months from the start month makes it 60 months or 5 years
    valid_df = df[df['월별'] > start_month + pd.DateOffset(months=59)]
    
    # Check if there are enough data points for validation
    if valid_df.empty:
        break
    
    # Train the AR(4) model on the current training set
    model = AutoReg(train_df['총인구'], lags=4)
    model_fit = model.fit()
    
    # Generate predictions on the validation set
    start = len(train_df)
    end = start + len(valid_df) - 1
    predictions = model_fit.predict(start=start, end=end)
    
    # Store the predictions in the DataFrame
    column_name = (start_month + pd.DateOffset(months=60)).strftime('%Y-%m')
    rolling_predictions_corrected[column_name] = predictions.reset_index(drop=True)
    
    # Update the start month for the next iteration
    start_month += pd.DateOffset(months=1)

# Show the first few rows of the DataFrame
rolling_predictions_corrected.head()


rolling_predictions_shifted_corrected = rolling_predictions_corrected.copy()
for i, col in enumerate(rolling_predictions_shifted_corrected.columns[1:], start=1):
    rolling_predictions_shifted_corrected[col] = rolling_predictions_shifted_corrected[col].shift(i)

# Show the first few rows of the DataFrame after shifting
rolling_predictions_shifted_corrected.head()











from statsmodels.tsa.ar_model import AutoReg
import numpy as np

# 원본 데이터를 날짜 인덱스와 함께 저장
df.set_index('월별', inplace=True)

# 결과를 저장할 빈 데이터 프레임 생성
result_df = pd.DataFrame(index=pd.date_range(start='2019-03-01', end='2023-04-01', freq='MS'))

# Train set의 시작과 끝을 설정
train_start = '2014-03-01'
train_end = '2019-02-01'

while True:
    # Train set과 Test set 분리
    train = df[train_start:train_end]
    test_len = len(df) - len(train)
    
    # 모델 적합
    model = AutoReg(train['총인구'], lags=4)
    model_fitted = model.fit()
    
    # 예측
    predictions = model_fitted.predict(start=len(train), end=len(train)+test_len-1, dynamic=False)
    
    # 예측 결과를 결과 데이터프레임에 추가
    col_name = f"Predict_from_{train_start}_to_{train_end}"
    result_df[col_name] = predictions
    
    # Train set의 시작과 끝을 1달씩 미룸
    train_start = pd.to_datetime(train_start) + pd.DateOffset(months=1)
    train_end = pd.to_datetime(train_end) + pd.DateOffset(months=1)
    
    # 더 이상 예측할 데이터가 없으면 반복문 종료
    if train_end >= df.index[-1]:
        break

# 결과 확인
result_df.head()










