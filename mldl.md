---
layout: archive-dates
permalink: /mldl/
title: Portfolio of Machine Learning(ML), Deep Learning(DL), Time Series(TS) and NLP Projects
---

In the following links, you can check out some interesting Machine learning, Deep Learning and Time series forecasting models I had worked with!

## ML & DL Models
-------------

### LeNet-5 CNN Network Implementation

The architecture of the LeNet-5 Network can be seen on the following visual:

<img src="/images/lenet5.jpeg?raw=true"/>

As seen on the visual, the LeNet-5 architecture consists of *two convolutional and average pooling layers*, followed by a *flattening convolutional layer*. After these layers, it has *two fully-connected layers* and finally a *softmax* classifier. In the following link, it can be seen the implementation in PyTorch.

- [LeNet 5 CNN Network](/Notebooks/LeNet5_CNN.html)


### Prediction of House Prices [(Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Many different methods (KNN Regressor, Decision Trees, Extra trees, Random forest, HistGradient Regressor and XGBoost) were implemented to predict the house prices. The best optimal model was chosen by doing cross validation with 3 folds. To improve the performance of each model, different techniques were implemented for hyper-parameter tuning such as Grid search, Random search, Bayes search, Optuna, Halving Grid Search and Halving Random Search.

- [Kaggle ML Prediction Competition](/Notebooks/Kaggle_Comp.html)

<img src="/images/ml1.PNG?raw=true"/>

### Gaussian Processes

To see the implementation of Gaussian processes with GPy, click on the following link:

- [Regression with Gaussian Processes](/Notebooks/GPs.md)


### Comparison of different kernel methods for multiclass classification

In this study, five different schema were implemented for face recognition classification. 

- [Kernel Methods](/Notebooks/kernels.html)


## Time Series Forecasting Statistical & ML Models
-------------

Here, I have placed some projects about time series forecasting with statistical and Machine learning models.

### Prediction of hourly traffic intensity in Madrid city

In this project, the traffic insensity level of the most crowded district of Madrid city was estimated in every hour by implementing both statistical and Machine Learning models. The data used in this project was obtained from the sensors placed in Madrid city.

- [Hourly Traffic Intensity Forecasting in Madrid](Notebooks/hourly_traffic_pred.html)

### Time Series Forecasting of the annual exports of Turkey

In this study, the annual exports of Turkey has been analyzed and exports has been forecasted by statistical tools such as ETS.

- [Annual exports of Turkey](Notebooks/turkey_annual_exports.html)

## Natural Language Processing Topic Model Prediction

In the following link, you can check out the project related with topic models prediction in python!

- [Topic models Prediction](/Notebooks/Topic_models.html)
