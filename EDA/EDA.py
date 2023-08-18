import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# plot 한글 안깨지게
from matplotlib import font_manager, rc
font_path = "C:/Users/jhmok/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


df = pd.read_excel("./data/Hwaseong_data.xlsx")



### 총인구 ###
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,2], label = df.columns[2])
plt.legend()
plt.title("총인구")
plt.show()


###### 정의상 수도권 + 지방권 = 전국 일테니, 전국이 수도권과 지방권의 평균처럼 나옴
###### 수도권이랑 화성이랑 비교하는게 나을듯
###### Fluctuation 있는 시기가 비슷함 (22년 이후?)


### 지가변동률 ###

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[18,19,20]], label = df.columns[[18,19,20]])
plt.legend()
plt.title("지가변동률(전국, 수도권, 지방)")
plt.show()

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[18,19,20,24]], label = df.columns[[18,19,20,24]])
plt.legend()
plt.title("지가변동률(전국, 수도권, 지방, 화성시)")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[19,24]], label = df.columns[[19,24]])
plt.legend()
plt.title("지가변동률(수도권, 화성시)")
plt.show()  # 22년 7월에 급락





### 매매가격지수 ### 
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[25,26,27,33]], label = df.columns[[25,26,27,33]])
plt.legend()
plt.title("매매가격지수(전국, 수도권, 지방, 화성시)")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[26,33]], label = df.columns[[26,33]])
plt.legend()
plt.title("매매가격지수(수도권, 화성시)")
plt.show()

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[32,33]], label = df.columns[[32,33]])
plt.legend()
plt.title("매매가격지수(서울, 화성시)")
plt.show()  # 22년 1월에 급락락








### 매매가격변동률 ###
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[35,42]], label = df.columns[[35,42]])
plt.legend()
plt.title("매매가격변동률(수도권, 화성시시)")
plt.show()

fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[41,42]], label = df.columns[[41,42]])
plt.legend()
plt.title("매매가격변동률(서울, 화성시시)")
plt.show()  # 21년 10월에 급락하다가 23년 3월에 올라감감





### 전세가격지수 ### 
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[44,51]], label = df.columns[[44,51]])
plt.legend()
plt.title("전세가격지수(수도권, 화성시)")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[50,51]], label = df.columns[[50,51]])
plt.legend()
plt.title("전세가격지수(서울, 화성시)")
plt.show()


### 월세가격지수 ###
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[62,69]], label = df.columns[[62,69]])
plt.legend()
plt.title("월세가격지수(수도권, 화성시)")
plt.show()


fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,[68,69]], label = df.columns[[68,69]])
plt.legend()
plt.title("월세가격지수(서울, 화성시)")
plt.show()


### 순이동 ###
fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[95:,0], df.iloc[95:,[81]], label = df.columns[[81]])
plt.legend()
plt.title("순이동")
plt.show() #9월부터 증가하긴함


###### 인구수는 어차피 우상향 추세이기 때문에, 주택가격지표들과 순이동과의 연관성을 보려했으나 잘 모르겠음...
###### 순이동을 한 달을 기준으로 봤을 때, 단위가 크지 않기 때문에 변동성이 커보이긴 하나 큰 의미없는 노이즈 때문일수도 있음







fig, ax = plt.subplots(figsize =(6, 4))
plt.plot(df.iloc[:,0], df.iloc[:,82], label = df.columns[82])
plt.legend()
plt.title("출생자수")
plt.show()
