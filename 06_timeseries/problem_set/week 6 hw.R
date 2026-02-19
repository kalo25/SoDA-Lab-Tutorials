###############################################################################
#Week 6 HW--Time Series (Soda 501)
#Kawain Lo
##############################################################################

# --- 0) Setup
set.seed(123)
# --- 1) Create a synthetic daily time series (trend + weekly seasonality + AR(1) noise)
n <- 600
dates <- seq.Date(from = as.Date("2024-01-01"), by = "day", length.out = n)
t <- 1:n


#PROMPT 1: Decomposition trend + seasonality + residual (save a figure). Extend 
#the synthetic daily time series in timeseries.R by decomposing it into components.

#convert the daily series to a ts object with weekly frequency (frequency = 7)
trend <- 0.02 * t
weekly <- 1.2 * sin(2 * pi * t / 7)


phi <- 0.65 #adding this violates IID 
eps <- rnorm(n, mean = 0, sd = 1.0) ##error term
ar_noise <- rep(NA_real_, n)
ar_noise[1] <- eps[1]
for (i in 2:n) {
  ar_noise[i] <- phi * ar_noise[i - 1] + eps[i] ##creates autocorrelation noise
}

y <- 10 + trend + weekly + ar_noise

df <- data.frame(
  date = dates,
  t = t,
  y = y
)

df_mod <-- ts(df, frequency=7)

#run a seasonal-trend decomposition (code "stl()")
#performing STL decomposition

df_decomposed <- stl(df_mod[, 1], s.window = "periodic")

#plot the decomposition

install.packages("tsibble")
library(ggplot2)
library(tsibble)

autoplot(df_decomposed) + labs(title = "Decomposition Plot")

#create a single figure showing the observed series and the decomposed components

par(mfrow = c(1, 2))
plot(df$date, df$y, type = "l", 
     main = "Synthetic daily time series: trend + weekly seasonality + AR(1) noise", 
     xlab = "Date", ylab = "y")
plot(autoplot(df_decomposed) + labs(title = "Decomposition Plot"), type='l')


p1 <- ggplot(df, aes(x = date, y = y)) + 
  geom_line() + 
  theme_classic() + 
  geom_smooth(se = F, col = "red") + 
  labs(x = "Date", y = "Outcome")

p2 <- autoplot(df_decomposed) + theme_minimal()

library(patchwork)

ts_plot <- p1 + p2 + plot_layout(ncol = 2)

#save the figure as outputs/figures/decomposition.png

ggsave("C:/Users/karra/Desktop/Coding_work/soda_501/06_timeseries/problem_set/outputs/figures/decomposition.png", plot = ts_plot, width = 10, height = 5)


#PROMPT 2: Rolling-origin evaluation (backtesting) for forecasting RMSE. The lecture
#emphasizes that evaluation should mimic deployment (fit on past, predict future). Implement
#a rolling-origin backtest that produces MANY out-of-sample errors instead of a single split

#choose a forecast horizon h = 1 day ahead
n1 <- length(df)
h <- 1

#choose an initial training window (i.e., first 300 days)
initial_window <- 300

error <- rep(NA_real_, n1)

#for each time t from the end of the initial window to the end of the series:
#fit an AR(1) model to data up to t (use arima(..., order = c(1,0,0)) or forecast::auto.arima if available

for (t in initial_window:(n1 - h)) {
  train_y <- y[1:t]
  fit <- arima(train_y, order = c(1, 0, 0))
  pred_1 <- predict(fit, n.ahead = h)$pred[h] #forecast y_t+1
    }
pred_t <- pred_1
#store the one-step-ahead error as e_t+1 = (y_t+1 - y-hat_t+1)
error[t + 1] <- y[t + 1] - pred_1
print(error[t+1])
error_final <- na.omit(error[t+1])
print(error[t+1])

#compute RMSE across the backtest errors and compare it to the single time-split RMSE from the tutorial
#single time-split RMSE from tutorial
cut <- n - test_n
train_idx_time <- 1:cut
test_idx_time  <- (cut + 1):n

y_train_time <- df$y[train_idx_time]
y_test_time  <- df$y[test_idx_time]

if (use_forecast) {
  fit_time <- forecast::auto.arima(y_train_time)
  pred_time <- as.numeric(forecast::forecast(fit_time, h = length(y_test_time))$mean)
} else {
  fit_time <- arima(y_train_time, order = c(1,0,0))
  pred_time <- as.numeric(predict(fit_time, n.ahead = length(y_test_time))$pred)
}

single_rmse <- sqrt(mean((y_test_time - pred_time)^2))

#backtest error RMSE
backtest_rmse <- sqrt(mean(error[t+1]^2))

print(single_rmse)
print(backtest_rmse)

#save a CSV of the backtest errors (date, y, y-hat, error) as "outputs/tables/backtest_errors.csv"
write.csv(backtest_rmse, "C:/Users/karra/Desktop/Coding_work/soda_501/06_timeseries/problem_set/outputs/tables/backtest_errors.csv")

#save a line plot of y_t and y-hat_t over the test region as "outputs/figures/backtest_forecast.png"
test_idx <- (initial_window + 1):n1

png("C:/Users/karra/Desktop/Coding_work/soda_501/06_timeseries/problem_set/outputs/figures/backtest_forecast.png", width=600, height=350)

plot(test_idx,
     y[test_idx], type = "l", col = "black",
     lwd = 2, xlab = "Time", ylab = "y",
     main = "Rolling-Origin: Actual vs Forecast (Test Region)")

lines(test_idx, (pred_t)[test_idx], col = "red",lwd = 2)

legend("topleft",
       legend = c("Actual y_t", "Forecast y-hat_t"),
       col = c("black", "red"),
       lwd = 2,
       bty = "n")

dev.off()

#PROMPT 3: Interrupted Time Series (ITS): level and slope change + placebo date
#create a synthetic intervention at time t0 on top of a trend (use the lecture's ITS framing)
set.seed(123)
n2 <- 300
t2 <- 1:n
t0 <- 150 #pick an intervention date t0 around the middle of series and report your choice


trend <- 0.02 * t2
weekly <- 1.2 * sin(2 * pi * t2 / 7)

alpha <- 5
delta <- 0.03
phi2 <- 0.75
u <- rnorm(n2, mean = 0, sd = 1)

e <- rep(NA_real_, n2)
e[1] <- u[1]
for (i in 2:n2) {
  e[i] <- phi2 * e[i - 1] + u[i]
}

y2 <- alpha + delta * t2 + e

#create an intervention indicator I[t >= t0] and a post-intervention time counter (t-t0) I[t>=t0]
intervention <- as.numeric(t2 >= t0)
post_intervention <- (t2 - t0) * intervention

#fit an ITS regression:
#y_t = alpha + delta_t + T1 I[t>=t0] + T2(t=t0) I[t>=t0] + epsilon_t

its_reg <- lm(y2 ~ t2 + intervention + post_intervention)

y_fitted <- fitted(its_reg)

coef_est <- coef(its_reg)

alpha_hat  <- coef_est["(Intercept)"]
delta_hat  <- coef_est["t"]
y_counterfactual <- alpha_hat + delta_hat * t2


#plot 3 lines on the same figure: observed y_t, fitted values from the ITS model, 
#counterfactual values setting T1=T2=0 

png("C:/Users/karra/Desktop/Coding_work/soda_501/06_timeseries/problem_set/outputs/figures/its_plot.png", width=600, height=350) #save main ITS figure as outputs/figures/its_plot.png

plot(t2, y2,
     type = "l", lwd = 2, col = "black",
     xlab = "Time", ylab = "y",
     main = "Interrupted Time Series with Counterfactual")

lines(t2, y_fitted, col = "red",
      lwd = 2)

lines(t2, y_counterfactual, col = "blue",
      lwd = 2,
      lty = 2)

abline(v = t0, lty = 3)

legend("topleft",
       legend = c("Observed y_t",
                  "Fitted ITS",
                  "Counterfactual (No Intervention)",
                  "Intervention Time"),
       col = c("black", "red", "blue", "black"),
       lwd = c(2,2,2,1),
       lty = c(1,1,2,3),
       bty = "n")

dev.off()

#run a placebo ITS with a FAKE intervention date in the pre-period and report 
#whether it produces a large effect + interpret results

t0_real     <- 150
t0_placebo  <- 40

vars <- function(t2, t0_real) {
  placebo <- as.numeric(t2 >= t0_real)
  post_time <- (t2 - t0_real) * placebo
  data.frame(df_placebo = placebo, post_time = post_time, t2 = t2, y2 = y2)
}

its_reg2 <- lm(y2 ~ t2 + intervention + post_intervention) #real intervention

its_vars_placebo <-vars(t2, t0_placebo)
reg_placebo <- lm(y2 ~ t2 + df_placebo + post_time, data = its_vars_placebo) #placebo regression

coef_real <- summary(its_reg2)$coefficients
coef_placebo <- summary(reg_placebo)$coefficients

##INTERPRETATION: the effect sizes are extremely large, to the point where I would be suspicious if this model was 
#run using real-life data. the real treatment model has an intervention coef of 0.847 and a post-intervention coef of 0.008, both of which are statistically significant
#the placebo has a coef of -0.085 and a time 2 (post "intervention") coef of 0.005, neither of which are statistically significant.
#this strongly suggests that the treatment effect does exist and has observable, major impacts on the model outcome.

#save the coefficient table for the real ITS and placebo ITS as outputs/tables/its_results.csv

library(dplyr)

coef_real <- as.data.frame(summary(its_reg2)$coefficients) %>% mutate(model = "Real ITS")
coef_placebo <- as.data.frame(summary(reg_placebo)$coefficients) %>% mutate(model = "Placebo ITS")

df_combined <- bind_rows(coef_real, coef_placebo)

write.csv(df_combined, "C:/Users/karra/Desktop/Coding_work/soda_501/06_timeseries/problem_set/outputs/tables/its_results.csv", row.names = TRUE)
