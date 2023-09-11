import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.stattools import adfuller

def check_assumption(f1, f2, y, loss_type = "SE") :
    # Calculate errors
    e1 = f1 - y
    e2 = f2 - y
    
    # Calculate loss based on specified loss type
    if loss_type == "SE":
        g1 = e1 ** 2
        g2 = e2 ** 2
    elif loss_type == "AE":
        g1 = np.abs(e1)
        g2 = np.abs(e2)
    elif loss_type == "SPE":
        g1 = ((y - f1) / f1) ** 2
        g2 = ((y - f2) / f2) ** 2
    elif loss_type == "ASE":
        g1 = np.abs(e1[1:]) / np.mean(np.abs(np.diff(y)))
        g2 = np.abs(e2[1:]) / np.mean(np.abs(np.diff(y)))
    elif isinstance(loss_type, (int, float)):
        g1 = np.exp(loss_type * e1) - 1 - loss_type * e1
        g2 = np.exp(loss_type * e2) - 1 - loss_type * e2
        
    data = g1 - g2
    result = adfuller(data)
    adf_statistic, p_value = result[0], result[1]
    if p_value <= 0.05:
        print(f"기본 가정을 만족하는 적합한 검정입니다.")
    else:
        print(f"기본 가정을 만족하지 않는 적절하지 않은 검정입니다.")

def DM_test(f1, f2, y, loss_type="SE", h=1, c=False, H1="same"):
    # Calculate errors
    e1 = f1 - y
    e2 = f2 - y
    
    # Calculate loss based on specified loss type
    if loss_type == "SE":
        g1 = e1 ** 2
        g2 = e2 ** 2
    elif loss_type == "AE":
        g1 = np.abs(e1)
        g2 = np.abs(e2)
    elif loss_type == "SPE":
        g1 = ((y - f1) / f1) ** 2
        g2 = ((y - f2) / f2) ** 2
    elif loss_type == "ASE":
        g1 = np.abs(e1[1:]) / np.mean(np.abs(np.diff(y)))
        g2 = np.abs(e2[1:]) / np.mean(np.abs(np.diff(y)))
    elif isinstance(loss_type, (int, float)):
        g1 = np.exp(loss_type * e1) - 1 - loss_type * e1
        g2 = np.exp(loss_type * e2) - 1 - loss_type * e2
    
    # Calculate d and other statistics
    d = g1 - g2
    T = len(d)
    dbar = np.mean(d)
    
    def gammahat(k):
        temp1 = d - dbar
        temp2 = np.concatenate([np.full(abs(k), np.nan), temp1])[:T]
        temp2 -= dbar
        temp = temp1 * temp2
        temp = temp[abs(k):]
        return np.sum(temp) / T
    
    # Calculate gdk
    if h > 1:
        gdk = np.array([gammahat(k) for k in range(1, h)])
    else:
        gdk = 0
    
    gdk = gammahat(0) + 2 * np.sum(gdk)
    
    # Calculate DM statistic
    DM = dbar / np.sqrt(gdk / T)
    
    # Calculate p-value based on alternative hypothesis
    if H1 == "same":
        pval = 2 * min(norm.sf(DM), norm.cdf(DM))
    elif H1 == "less":
        pval = norm.sf(DM)
    elif H1 == "more":
        pval = norm.cdf(DM)
        
    # Apply Harvey-Leybourne-Newbold correction for small samples if c=True
    if c:
        DM *= np.sqrt((T + 1 - 2 * h + h * (h - 1)) / T)
        if H1 == "same":
            pval = 2 * min(1 - t.cdf(DM, df=T - 1), t.cdf(DM, df=T - 1))
        elif H1 == "less":
            pval = 1 - t.cdf(DM, df=T - 1)
        elif H1 == "more":
            pval = t.cdf(DM, df=T - 1)
    
    # Formulate alternative hypothesis statement
    if H1 == "same":
        alt = "Forecast f1 and f2 have different accuracy."
    elif H1 == "less":
        alt = "Forecast f1 is less accurate than f2."
    elif H1 == "more":
        alt = "Forecast f1 is more accurate than f2."
    
    
    result = {
        "statistic": DM,
        "parameter": h,
        "alternative": alt,
        "p_value": pval,
        "method": "Diebold-Mariano test",
        "data_name": f"f1 and f2 and y"
    }
    
    return result




df = pd.read_excel("./data/최종 예측값 테이블.xlsx")
df_sort = df.sort_values(by = "RMSE")

df0 = np.array(df_sort.iloc[-1, 1:13])
df1 = np.array(df_sort.iloc[0, 1:13])
df2 = np.array(df_sort.iloc[1, 1:13])



result = DM_test(df1, df2, df0, loss_type = "SE", h=1, c=True, H1="more")
check_assumption(df1, df2, df0)
print("p-value", result["p_value"])
