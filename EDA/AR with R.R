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


dif <- diff(diff(df_ts))

fit <- Arima(dif, order = c(4,0,0))
checkresiduals(fit)
forecast(fit)
autoplot(forecast(fit))


fit <- Arima(diff(log(df_ts)), order = c(4,0,0))
checkresiduals(fit)
forecast(fit)

a <- forecast(fit)


fit <- Arima(df_ts, order = c(4,1,0))
checkresiduals(fit)

fit <- Arima(df_ts, order = c(0,2,0))
checkresiduals(fit)

autoplot(forecast(fit))





model <- auto.arima(df_ts)
autoplot(forecast(model, h =  32))




df <- read_xlsx("./data/찐 최종 데이터.xlsx")
df_ts <- ts(df$총인구, start = 2014, end = 2022, frequency = 12)
fit <- Arima(df_ts, order = c(1,0,0))
forecast(fit)
result <- forecast(fit)


rs <- as.numeric(result$mean); rs <- rs[1:15]



a <- df[seq(96,110),]
a <- as.numeric(unlist(a[,2]))



sqrt(mean((a-rs)^2))



plot(rs, ylim = c(-2000,2000), type = "l", col = "blue")
lines(a, type = "l", col = "red")
