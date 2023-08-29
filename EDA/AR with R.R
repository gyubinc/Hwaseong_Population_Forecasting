library(readxl)
library(tidyverse)
library(ggplot2)
library(forecast)
library(fpp3)
library(fpp2)

df <- read_xlsx("./data/Hwaseong_data.xlsx")
df <- df %>% select(c("월별","총인구"))
df$월별 <- as.Date(df$월별)
df_ts <- ts(df$총인구, start = 2014, frequency = 12)

ggtsdisplay(df_ts)

df_ts %>% diff %>% ggtsdisplay(main="1차 차분")

df_ts %>% diff %>% diff %>% ggtsdisplay(main="2차 차분")


fit <- Arima(diff(log(df_ts)), order = c(4,0,0))
checkresiduals(fit)
forecast(fit)

a <- forecast(fit)


fit <- Arima(df_ts, order = c(4,1,0))
checkresiduals(fit)

fit <- Arima(df_ts, order = c(4,2,0))
checkresiduals(fit)

autoplot(forecast(fit))
