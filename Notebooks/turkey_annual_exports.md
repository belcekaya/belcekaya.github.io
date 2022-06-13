---
layout: archive-dates
title: Time Series Forecasting of Annual Exports of Turkey
toc: true
---

# Part 1

Data set *globaleconomy* contains the annual Exports from many
countries.

GDP: Gross domestic product (in $USD February 2019)  
Growth: Annual percentage growth in GDP  
CPI: Consumer price index (base year 2010)  
Imports: Imports of goods and services (% of GDP)  
Exports: Exports of goods and services (% of GDP)  
Population: Total population

Each series is uniquely identified by one key: -&gt; Country: The
country or region of the series.

Among all countries, one selected country will be analyzed.

    library(fpp2)
    library(tsibbledata)
    head(global_economy)

    ## # A tibble: 6 x 9
    ##   Country     Code   Year         GDP Growth   CPI Imports Exports Population
    ##   <fct>       <fct> <dbl>       <dbl>  <dbl> <dbl>   <dbl>   <dbl>      <dbl>
    ## 1 Afghanistan AFG    1960  537777811.     NA    NA    7.02    4.13    8996351
    ## 2 Afghanistan AFG    1961  548888896.     NA    NA    8.10    4.45    9166764
    ## 3 Afghanistan AFG    1962  546666678.     NA    NA    9.35    4.88    9345868
    ## 4 Afghanistan AFG    1963  751111191.     NA    NA   16.9     9.17    9533954
    ## 5 Afghanistan AFG    1964  800000044.     NA    NA   18.1     8.89    9731361
    ## 6 Afghanistan AFG    1965 1006666638.     NA    NA   21.4    11.3     9938414

## All countries

    library(dplyr)
    #Filter for Exports for one country
    global_economy %>% 
      distinct(Country)

    ## # A tibble: 263 x 1
    ##    Country            
    ##    <fct>              
    ##  1 Afghanistan        
    ##  2 Albania            
    ##  3 Algeria            
    ##  4 American Samoa     
    ##  5 Andorra            
    ##  6 Angola             
    ##  7 Antigua and Barbuda
    ##  8 Arab World         
    ##  9 Argentina          
    ## 10 Armenia            
    ## # ... with 253 more rows

I decided to choose as country “Turkey”.

    #Filter for Exports for one country
    global_economy %>% 
        filter(Country == 'Turkey') -> tr_global_economy
    tr_global_economy

    ## # A tibble: 58 x 9
    ##    Country Code   Year          GDP Growth       CPI Imports Exports Population
    ##    <fct>   <fct> <dbl>        <dbl>  <dbl>     <dbl>   <dbl>   <dbl>      <dbl>
    ##  1 Turkey  TUR    1960 13995067818.  NA    0.0000540    3.67    2.06   27472331
    ##  2 Turkey  TUR    1961  8022222222.   1.16 0.0000557    6.79    5.12   28146893
    ##  3 Turkey  TUR    1962  8922222222.   5.57 0.0000578    7.97    5.60   28832805
    ##  4 Turkey  TUR    1963 10355555556.   9.07 0.0000615    6.97    4.18   29531342
    ##  5 Turkey  TUR    1964 11177777778.   5.46 0.0000622    5.47    4.47   30244232
    ##  6 Turkey  TUR    1965 11944444444.   2.82 0.0000650    5.40    4.56   30972965
    ##  7 Turkey  TUR    1966 14122222222.  11.2  0.0000705    5.66    4.09   31717477
    ##  8 Turkey  TUR    1967 15666666667.   4.73 0.0000804    4.96    4.11   32477961
    ##  9 Turkey  TUR    1968 17500000000    6.78 0.0000853    5.08    3.68   33256432
    ## 10 Turkey  TUR    1969 19466666667.   4.08 0.0000895    4.74    3.60   34055361
    ## # ... with 48 more rows

## a) Plotting the Exports series

### Defining the Year and Exports columns as a time series data

    ts_tr_global_economy <- ts(tr_global_economy$Exports, 
                               start = min(tr_global_economy$Year), frequency = 1)
    ts_tr_global_economy

    ## Time Series:
    ## Start = 1960 
    ## End = 2017 
    ## Frequency = 1 
    ##  [1]  2.055800  5.124654  5.603985  4.184549  4.473161  4.558140  4.091267
    ##  [8]  4.113475  3.682540  3.595890  4.427481  5.319588  6.018679  7.032967
    ## [15]  5.728116  4.421347  4.859086  3.815204  4.146912  3.218027  5.161932
    ## [22]  8.236932 11.864051 12.473213 15.606603 15.860723 13.312438 15.580840
    ## [29] 18.653970 16.202916 13.365103 13.841130 14.392236 13.673803 21.362131
    ## [36] 19.891608 21.542658 24.581718 20.568308 18.577919 19.448739 26.577895
    ## [43] 24.460686 22.243140 22.750501 21.017779 21.650406 21.220222 22.826254
    ## [50] 22.573669 20.448807 22.262411 23.667368 22.272256 23.764333 23.345935
    ## [57] 21.965081 24.804230

    autoplot(ts_tr_global_economy, xlab="Year",ylab="Exports")

![](turkey_annual_exports_files/figure-markdown_strict/unnamed-chunk-5-1.png)
The series is representing the yearly exports data and in the series,
there is an upward trend throughout the years, but there are not certain
seasonal patterns, therefore the time series is not seasonal. The trend
has variance and the series shows multiplicative trend and changing
levels rules out series.

Between 1960 and 1980, the number of exports were more or less similar
although there was an increase in year 1970. After 1980, there was a
sharp increase in exports until 1990. In 1990, the exports went down for
some years, then the number of exports started to increase again and the
series of exports had an upward trend in the last years.

### Spliting the data into train and test

Here, first the 46 years (which is the 80% of the all rows of the data)
will be assigned to the train series and the rest which is the last 12
years will be given to the test series. Therefore for the train, the
data between 1960 and 2005 will be taken and for the test set from 2006
to 2017. While doing predictions, the aim will be to predict the next 12
years, this two-year period.

    dim <- nrow(tr_global_economy)

    # for train, the first 46 years will be taken.
    floor(dim*0.80)

    ## [1] 46

    # for test, the last 12 years will be taken.
    dim-floor(dim*0.80)

    ## [1] 12

    ## 1960 - 2005
    train <- window(ts_tr_global_economy, end = 2005)
    test <- window(ts_tr_global_economy,  start= 2006)
    plot(ts_tr_global_economy)
    lines(train,col="red")
    lines(test, col="blue")

![](turkey_annual_exports_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    train

    ## Time Series:
    ## Start = 1960 
    ## End = 2005 
    ## Frequency = 1 
    ##  [1]  2.055800  5.124654  5.603985  4.184549  4.473161  4.558140  4.091267
    ##  [8]  4.113475  3.682540  3.595890  4.427481  5.319588  6.018679  7.032967
    ## [15]  5.728116  4.421347  4.859086  3.815204  4.146912  3.218027  5.161932
    ## [22]  8.236932 11.864051 12.473213 15.606603 15.860723 13.312438 15.580840
    ## [29] 18.653970 16.202916 13.365103 13.841130 14.392236 13.673803 21.362131
    ## [36] 19.891608 21.542658 24.581718 20.568308 18.577919 19.448739 26.577895
    ## [43] 24.460686 22.243140 22.750501 21.017779

    test

    ## Time Series:
    ## Start = 2006 
    ## End = 2017 
    ## Frequency = 1 
    ##  [1] 21.65041 21.22022 22.82625 22.57367 20.44881 22.26241 23.66737 22.27226
    ##  [9] 23.76433 23.34594 21.96508 24.80423

## b) Using an ETS(A,N,N) model to forecast the series, and ploting the forecasts.

### Model Fitting

In this step, the ETS(A,N,N) model will be fit the data. ETS model
combines a “level”, “trend” (slope) and “seasonal” component to describe
a time series. Therefore, in this case, the model of ETS has the
Additive Error, None Trend and None Seasonal parameters. This model is a
simple exponential smoothing predictive model, therefore, it estimates
the *α* parameter which controls the flexibility of the level. The *α*
parameter places between 0 and 1 and it is a smoothing parameter for the
simple exponential smoothing. This model type would be suitable for
forecasting the Exports series of Turkey because in the series there is
no clear trend(trend has variance) or seasonal pattern.

    ets_ann <- ets(train,model="ANN")
    summary(ets_ann)

    ## ETS(A,N,N) 
    ## 
    ## Call:
    ##  ets(y = train, model = "ANN") 
    ## 
    ##   Smoothing parameters:
    ##     alpha = 0.954 
    ## 
    ##   Initial states:
    ##     l = 2.1961 
    ## 
    ##   sigma:  2.388
    ## 
    ##      AIC     AICc      BIC 
    ## 260.1545 260.7260 265.6404 
    ## 
    ## Training set error measures:
    ##                     ME     RMSE      MAE      MPE     MAPE      MASE
    ## Training set 0.4306997 2.335506 1.670888 2.640637 15.52892 0.9748054
    ##                     ACF1
    ## Training set -0.03027617

As seen on the above results,

-   The AIC has been found 260.15 and BIC 265.64 for ETS(A,N,N) model.

-   The RMSE training error has been found 2.3355.

-   Smoothing parameter *α* has been estimated close to 1, it indicates
    the learning is fast because with the high *α* value, more weight
    has placed on the most recent observations when making forecasts of
    future values. Therefore, recent changes in the data will have a
    bigger impact on predicted values.

### Checking the residuals of the ETS(A,N,N) model

Here, it is clear that the ETS(A,N,N) model’s residuals present white
noise, because the p-value of the Ljung-Box test is larger than 0.05. As
residuals are white noise, the residuals are stationary.

    checkresiduals(ets_ann)

![](turkey_annual_exports_files/figure-markdown_strict/unnamed-chunk-10-1.png)

    ## 
    ##  Ljung-Box test
    ## 
    ## data:  Residuals from ETS(A,N,N)
    ## Q* = 10.153, df = 7, p-value = 0.1801
    ## 
    ## Model df: 2.   Total lags used: 9

### Forecasting with ETS(A,N,N) the last 12 years

As seen from the point forecast values for each year, all forecasts are
constant. It happens because in the ETS model the components have been
assigned to no trend and no seasonality. That is why all of them are the
same value.

    forecast(ets_ann, h=12)

    ##      Point Forecast    Lo 80    Hi 80     Lo 95    Hi 95
    ## 2006       21.09666 18.03632 24.15700 16.416270 25.77704
    ## 2007       21.09666 16.86708 25.32623 14.628077 27.56523
    ## 2008       21.09666 15.95730 26.23601 13.236694 28.95662
    ## 2009       21.09666 15.18594 27.00737 12.056993 30.13632
    ## 2010       21.09666 14.50422 27.68909 11.014394 31.17892
    ## 2011       21.09666 13.88667 28.30664 10.069938 32.12337
    ## 2012       21.09666 13.31800 28.87531  9.200228 32.99308
    ## 2013       21.09666 12.78816 29.40515  8.389906 33.80341
    ## 2014       21.09666 12.29014 29.90317  7.628249 34.56506
    ## 2015       21.09666 11.81881 30.37450  6.907417 35.28589
    ## 2016       21.09666 11.37030 30.82301  6.221476 35.97184
    ## 2017       21.09666 10.94158 31.25174  5.565800 36.62751

    test

    ## Time Series:
    ## Start = 2006 
    ## End = 2017 
    ## Frequency = 1 
    ##  [1] 21.65041 21.22022 22.82625 22.57367 20.44881 22.26241 23.66737 22.27226
    ##  [9] 23.76433 23.34594 21.96508 24.80423

**Plotting the forecast**

    ets_ann %>% forecast(h=12) %>%
      autoplot() 

![](turkey_annual_exports_files/figure-markdown_strict/unnamed-chunk-12-1.png)

## c) Compute the RMSE values for the training and test series

    accuracy(forecast(ets_ann,h=12),test)

    ##                     ME     RMSE      MAE      MPE     MAPE      MASE
    ## Training set 0.4306997 2.335506 1.670888 2.640637 15.52892 0.9748054
    ## Test set     1.4700919 1.869499 1.578067 6.269156  6.79718 0.9206528
    ##                     ACF1 Theil's U
    ## Training set -0.03027617        NA
    ## Test set     -0.02358438   1.23039

Also, the RMSE value could be calculated individually by getting the
residuals of the ETS(A,N,N) model and calculating the root mean squared
value of the residuals.

    sqrt(mean(ets_ann$residuals^2))

    ## [1] 2.335506

## d) Comparing the results to those from an ETS(A,A,N) model

### Model Fitting

In this step, the ETS(A,A,N) model will be fit the data. In this case,
the model of ETS has the Additive Error, Additive Trend and None
Seasonal components. This model type uses Holt’s Method. This method
makes predictions for the time series with a trend using two smoothing
parameters, *α* and *β*, which correspond to the level and trend
components, respectively.

As additive trend has been assigned to the model, apart from estimating
*α* like simple exponential smoothing model, this ETS(A,A,N) model will
also estimate the *β* parameter. *β* parameter controls the flexibility
of the trend.

    ets_aan <- ets(train,model="AAN")
    summary(ets_aan)

    ## ETS(A,A,N) 
    ## 
    ## Call:
    ##  ets(y = train, model = "AAN") 
    ## 
    ##   Smoothing parameters:
    ##     alpha = 0.8823 
    ##     beta  = 1e-04 
    ## 
    ##   Initial states:
    ##     l = 1.9417 
    ##     b = 0.4204 
    ## 
    ##   sigma:  2.3983
    ## 
    ##      AIC     AICc      BIC 
    ## 262.4096 263.9096 271.5529 
    ## 
    ## Training set error measures:
    ##                         ME     RMSE      MAE       MPE     MAPE     MASE
    ## Training set -0.0001098715 2.291628 1.665735 -3.445951 16.68647 0.971799
    ##                    ACF1
    ## Training set 0.02647245

As seen on the above results,

-   The AIC has been found 262.40 and BIC 263.90 for ETS(A,N,N) model.

-   The RMSE training error has been found 2.2916.

-   Smoothing parameter *α* has been estimated close to 1, it indicates
    that more weight has placed on the most recent observations when
    making forecasts of future values. Therefore, recent changes in the
    data will have a bigger impact on predicted values.

-   Smoothing parameter *β* has been estimated close to 0 (calculated
    0.0001), it indicates that the estimated trend represents a linear
    regression trend and it also means slow learning for the trend.

### Checking the residuals of the ETS(A,A,N) model

Here, it is clear that the ETS(A,A,N) model’s residuals present white
noise, because the p-value of the Ljung-Box test is larger than 0.05. As
residuals are white noise, the residuals are stationary.

    checkresiduals(ets_aan)

![](turkey_annual_exports_files/figure-markdown_strict/unnamed-chunk-16-1.png)

    ## 
    ##  Ljung-Box test
    ## 
    ## data:  Residuals from ETS(A,A,N)
    ## Q* = 9.4015, df = 5, p-value = 0.09408
    ## 
    ## Model df: 4.   Total lags used: 9

## e) Comparing the forecasts from both methods

**Plotting the forecast**

    library(ggpubr)
    theme_set(theme_pubr())

    ann <- autoplot(forecast(ets_ann, h=12),xlab="Year") + 
      ggtitle("Simple Exponential Smoothing-ETS(A,N,N)")
    aan <- autoplot(forecast(ets_aan, h=12),xlab="Year")+ 
      ggtitle("Holt's method-ETS(A,A,N)")

    all_figures <- ggarrange(ann, aan,
                        labels = c("1", "2"),
                        ncol = 2, nrow = 1)
    all_figures

![](turkey_annual_exports_files/figure-markdown_strict/unnamed-chunk-17-1.png)

### Comparing the model statistics for the training and test series

**Model summaries**

Here, the summary model statistics will be analyzed for each model to
check AIC and BIC scores. Because the error measures on the training set
are not really suitable for model selection. It is because in the
training sample it is always possible to over fit. Therefore,
information criteria like AIC or BIC take this into account and penalize
for the model complexity accordingly. Therefore, for the training set,
those AIC and BIC scores generally are suitable for model selection.

    summary(ets_ann)

    ## ETS(A,N,N) 
    ## 
    ## Call:
    ##  ets(y = train, model = "ANN") 
    ## 
    ##   Smoothing parameters:
    ##     alpha = 0.954 
    ## 
    ##   Initial states:
    ##     l = 2.1961 
    ## 
    ##   sigma:  2.388
    ## 
    ##      AIC     AICc      BIC 
    ## 260.1545 260.7260 265.6404 
    ## 
    ## Training set error measures:
    ##                     ME     RMSE      MAE      MPE     MAPE      MASE
    ## Training set 0.4306997 2.335506 1.670888 2.640637 15.52892 0.9748054
    ##                     ACF1
    ## Training set -0.03027617

    summary(ets_aan)

    ## ETS(A,A,N) 
    ## 
    ## Call:
    ##  ets(y = train, model = "AAN") 
    ## 
    ##   Smoothing parameters:
    ##     alpha = 0.8823 
    ##     beta  = 1e-04 
    ## 
    ##   Initial states:
    ##     l = 1.9417 
    ##     b = 0.4204 
    ## 
    ##   sigma:  2.3983
    ## 
    ##      AIC     AICc      BIC 
    ## 262.4096 263.9096 271.5529 
    ## 
    ## Training set error measures:
    ##                         ME     RMSE      MAE       MPE     MAPE     MASE
    ## Training set -0.0001098715 2.291628 1.665735 -3.445951 16.68647 0.971799
    ##                    ACF1
    ## Training set 0.02647245

-   According to the AIC and BIC scores of the training series, the
    ETS(A,N,N) has smaller scores than ETS(A,A,N) model.
-   Indeed, as the trend of this time series has a lot of variance, the
    simple exponential smoothing method would be a better choose
    according to the training.
-   On the other hand, it should be considered that in the last years of
    the series, the exports are increasing, in other words there is an
    upward trend. In that case the Holt’s method would capture the
    general upward trend of Turkey exports while the simple exponential
    smoothing model captures the general level or average level of the
    exports.

However, it should be also considered evaluating the prediction
performance on the test, as a model which fits the data well does not
necessarily forecast well.

**Error measures of Train and Test**

    # 1.Model- ETS(A,N,N)
    accuracy(forecast(ets_ann,h=12),test)

    ##                     ME     RMSE      MAE      MPE     MAPE      MASE
    ## Training set 0.4306997 2.335506 1.670888 2.640637 15.52892 0.9748054
    ## Test set     1.4700919 1.869499 1.578067 6.269156  6.79718 0.9206528
    ##                     ACF1 Theil's U
    ## Training set -0.03027617        NA
    ## Test set     -0.02358438   1.23039

    # 2.Model- ETS(A,A,N)
    accuracy(forecast(ets_aan,h=12),test)

    ##                         ME     RMSE      MAE       MPE      MAPE      MASE
    ## Training set -0.0001098715 2.291628 1.665735 -3.445951 16.686468 0.9717990
    ## Test set     -1.4403773648 1.867472 1.488789 -6.466387  6.678473 0.8685675
    ##                    ACF1 Theil's U
    ## Training set 0.02647245        NA
    ## Test set     0.16925486  1.199866

After calculating the error measures for the test series, according to
the results (such as RMSE, MAE,MAPE), Holt’s method is better than the
simple exponential smoothing method with a few difference. *(For example
in test set; RMSE for ETS(A,N,N) is 1.8694 and for ETS(A,A,N) is
1.8674)*

## f) Calculating a 95% prediction interval for the first forecast for each model, using the RMSE values and assuming normal errors

### 95% prediction interval(1st forecast) for ETS(A,N,N)-Simple Exp Smoothing:

**Intervals using the RMSE values and assuming normal errors**

    pred_ann <- forecast(ets_ann,h=12)
    rmse_ann <- sqrt(mean(pred_ann$residuals^2)) ## rmse of residuals

    glue::glue('Lower interval: ',pred_ann$mean[1] - 1.95 * rmse_ann) #lower 95%

    ## Lower interval: 16.5424194391249

    glue::glue('Upper interval: ',pred_ann$mean[1] + 1.95 * rmse_ann) #upper 95%

    ## Upper interval: 25.6508920248868

**Calculated intervals with those produced using R**

    # lower
    down_95 <- data.frame(pred_ann$lower)
    glue::glue('Lower interval: ',down_95$X95.[1])

    ## Lower interval: 16.4162703190677

    # upper
    up_95 <- data.frame(pred_ann$upper)
    glue::glue('Upper interval: ',up_95$X95.[1])

    ## Upper interval: 25.777041144944

### 95% prediction interval(1st forecast) for ETS(A,A,N)-Holt’s Method Exp Smoothing:

**Intervals using the RMSE values and assuming normal errors**

    pred_aan <- forecast(ets_aan,h=12)
    rmse_aan <- sqrt(mean(pred_aan$residuals^2)) ## rmse of residuals

    glue::glue('Lower interval: ',pred_aan$mean[1] - 1.95 * rmse_aan) #lower 95%

    ## Lower interval: 17.226346741643

    glue::glue('Upper interval: ',pred_aan$mean[1] + 1.95 * rmse_aan) #upper 95%

    ## Upper interval: 26.163695383076

**Calculated intervals with those produced using R**

    # lower
    down_95 <- data.frame(pred_aan$lower)
    glue::glue('Lower interval: ',down_95$X95.[1])

    ## Lower interval: 16.9944951135434

    # upper
    up_95 <- data.frame(pred_aan$upper)
    glue::glue('Upper interval: ',up_95$X95.[1])

    ## Upper interval: 26.3955470111755

**Comparing the 95% prediction interval for the first forecast in
ETS(A,A,N)**

In the Simple exponential smoothing model, when the 95% intervals has
been calculated by using the RMSE values, the range of lower and upper
values got smaller than the intervals calculated by R. It is important
because if the range between lower and upper is smaller, it means that
the forecast range has less variation. It is a good thing. Therefore,
when the intervals calculated by RMSE values, the forecasting intervals
has improved for this model.

-   Intervals with RMSE values: Lower interval: 16.5424194391249, Upper
    interval: 25.6508920248868
-   Intervals produced by R: Lower interval: 16.4162703190677, Upper
    interval: 25.777041144944

In the Holt’s method exponential smoothing model, when the 95% intervals
has been calculated by using the RMSE values, again, the range of lower
and upper values are smaller than the intervals calculated by R. For
this case also the forecast range found with less variation. It is a
good thing. Therefore, when the intervals calculated by RMSE values, the
forecasting intervals has improved.

-   Intervals with RMSE values: Lower interval: 17.226346741643, Upper
    interval: 26.163695383076
-   Intervals produced by R: Holt: Lower interval: 16.9944951135434,
    Upper interval: 26.3955470111755

When comparing these two models’ intervals, it is clear that the Holt’s
method lower and upper interval values are larger than the simple
exponential smoothing one. For instance lower and upper interval values
are 16.54 and 25.65 for simple exponential smoothing model and 17.22 and
26.16 for the Holt’s method. It indicates again that as the Holt’s
method has a trend component and the prediction trend represents the
upward trend, the interval values are larger than the simple smoothing
exponential model’s ones.
