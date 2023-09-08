import pandas as pd
from scipy.stats import t
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt 

df1 = pd.DataFrame([-1033.137219,880.1831193,-760.8555032,-209.5581889,-378.7676186,-2495.966267,165.9144285,327.3453827,-405.4451365,-585.9153313,947.7145591,274.2871115
])
df2 = pd.DataFrame([1867.994141,-401.6391602,-1227.361084,462.5367126,-2267.583252,-2373.152954,338.1312866,63.38195801,194.2364502,-680.7015686,2787.735229,38.76818848
])


def gamma_k(df1, df2, k) :
    n = len(df1)
    d_bar = (df1**2 - df2**2).mean()[0]
    d = []
    for i in range(0, n) :
        diff = df1[0][i]**2 - df2[0][i]**2
        d.append(diff)
    
    gamma = 0
    for t in range(k, n) :
        gamma += ((d[t] - d_bar)*(d[t-k]- d_bar)) * 1/n
    return gamma



def var_d(df1, df2) :
    n = len(df1)
    h = int(n**(1/3) + 1)
    gamma_0 = gamma_k(df1, df2, 0)
    gamma_sum = 0  # Initialize the sum
    for k in range(1, h):
        gamma_sum += gamma_k(df1, df2, k)
    result = 1/n * (gamma_0 + 2*gamma_sum)
    return result



def test_statistic(df1, df2) :
    n = len(df1)
    h = int(n**(1/3) + 1)
    d_bar = (df1**2 - df2**2).mean()[0]
    cons = ((n + 1 - 2*h + 1/n*h*(h-1))/n)**(1/2)
    s1 = var_d(df1, df2)**(-1/2) * (df1**2 - df2**2).mean()[0]
    s1_star = cons*s1
    return s1_star


def MDM_test(df1, df2) :
    n = len(df1)
    p_value = t.cdf(test_statistic(df1, df2), n-1)
    if p_value < 0.05 :
        print(f"방법1의 예측 정확도가 방법2의 예측정확도보다 통계적으로 유의하게 높습니다.")
    else :
        print(f"방법1의 예측 정확도와 방법2의 예측정확도는 차이가 없습니다.")
        

def check_assumption(df1, df2) :
    data = df1**2 - df2**2 
    result = adfuller(data)
    adf_statistic, p_value = result[0], result[1]
    if p_value <= 0.05:
        print(f"기본 가정을 만족하는 적합한 검정입니다.")
    else:
        print(f"기본 가정을 만족하지 않는 적절하지 않은 검정입니다.")





    

