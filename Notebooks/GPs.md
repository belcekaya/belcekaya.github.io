---
layout: archive-dates
title: Regression with Gaussian Processes
toc: true
---

The aim of this project is to solve a real data problem using the Gaussian Process implementation of GPy. The documentation of GPy is avaialable from the [SheffieldML github page](https://github.com/SheffieldML/GPy) or from [this page](http://gpy.readthedocs.org/en/latest/). 

The problem is the prediction of both the heating load (HL) and cooling load (CL) of residential buildings. We consider eight input variables for each building: relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area, glazing area distribution.

In this [paper](https://www.sciencedirect.com/science/article/pii/S037877881200151X) you can find a detailed description of the problem and a solution based on linear regression [(iteratively reweighted least squares (IRLS) algorithm)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=10&ved=2ahUKEwjZuoLY2OjgAhUs3uAKHUZ7BVcQFjAJegQIAhAC&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F9b92%2F18e7233f4d0b491e1582c893c9a099470a73.pdf&usg=AOvVaw3YDwqZh1xyF626VqfnCM2k) and random forests. Using GPs, our goal is not only estimate accurately both HL and CL, but also get a measure of uncertainty in our predictions.

The data set can be downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency#).

### 1. Loading and preparing the data

* Download the dataset
* Divide at random the dataset into train (80%) and test (20%) datasets 

The data set consists of 768 instances and 8 attributes, which are:


* X1: Relative Compactness
* X2: Surface Area
* X3: Wall Area
* X4: Roof Area
* X5: Overall Height
* X6: Orientation
* X7: Glazing Area
* X8: Glazing Area Distribution
* y1: Heating Load
* y2: Cooling Load





```python
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_excel("/content/ENB2012_data.xlsx")
input_keys = ['X1','X2','X3','X4','X5','X6','X7','X8']
output_keys = ['Y1','Y2']
X = df[input_keys]
y = df[output_keys]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
```

### 2. Setting and optimizing the model

Two independent GPs will be trained, one to estimate HL and one to estimate CL. For each of the two GPs ...



```python
!pip install GPy 
import GPy
import numpy as np
import matplotlib.pyplot as plt

```

    Collecting GPy
      Downloading GPy-1.10.0.tar.gz (959 kB)
    [K     |████████████████████████████████| 959 kB 9.4 MB/s 
    [?25hRequirement already satisfied: numpy>=1.7 in /usr/local/lib/python3.7/dist-packages (from GPy) (1.21.6)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from GPy) (1.15.0)
    Collecting paramz>=0.9.0
      Downloading paramz-0.9.5.tar.gz (71 kB)
    [K     |████████████████████████████████| 71 kB 11.8 MB/s 
    [?25hRequirement already satisfied: cython>=0.29 in /usr/local/lib/python3.7/dist-packages (from GPy) (0.29.28)
    Requirement already satisfied: scipy>=1.3.0 in /usr/local/lib/python3.7/dist-packages (from GPy) (1.4.1)
    Requirement already satisfied: decorator>=4.0.10 in /usr/local/lib/python3.7/dist-packages (from paramz>=0.9.0->GPy) (4.4.2)
    Building wheels for collected packages: GPy, paramz
      Building wheel for GPy (setup.py) ... [?25l[?25hdone
      Created wheel for GPy: filename=GPy-1.10.0-cp37-cp37m-linux_x86_64.whl size=2565113 sha256=74c2dba8fbcb05f62c3e72120e64088af6a754653905065e0ab6fa91e5470999
      Stored in directory: /root/.cache/pip/wheels/f7/18/28/dd1ce0192a81b71a3b086fd952511d088b21e8359ea496860a
      Building wheel for paramz (setup.py) ... [?25l[?25hdone
      Created wheel for paramz: filename=paramz-0.9.5-py3-none-any.whl size=102566 sha256=65acd7dea17acea11621cc80c01c8e7b65462722f877ca83898d05bc3e240aec
      Stored in directory: /root/.cache/pip/wheels/c8/95/f5/ce28482da28162e6028c4b3a32c41d147395825b3cd62bc810
    Successfully built GPy paramz
    Installing collected packages: paramz, GPy
    Successfully installed GPy-1.10.0 paramz-0.9.5
    

## Operations on the training set

#### a) Building a GP regression model based on a RBF kernel with ARD, in which each input dimension is weighted with a different lengthscale.

#### b) Fitting the covariance function parameters and noise variance.


```python
### GP for HL
# define kernel
ker1 = GPy.kern.RBF(8,ARD=True) + GPy.kern.White(8)

# create simple GP model
m1 = GPy.models.GPRegression(X_train,y_train[['Y1']],ker1)

# optimize and plot
m1.optimize(messages=True,max_f_eval = 1000)

### GP for CL
# define kernel
ker2 = GPy.kern.RBF(8,ARD=True) + GPy.kern.White(8)

# create simple GP model
m2 = GPy.models.GPRegression(X_train,y_train[['Y2']],ker2)

# optimize and plot
m2.optimize(messages=True,max_f_eval = 1000)



```


    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…





    <paramz.optimization.optimization.opt_lbfgsb at 0x7fa073a4df90>



#### c) According to the ARD parameters found, finding the important variables for the regression. Comparig it to Table 8 in this [paper](https://www.sciencedirect.com/science/article/pii/S037877881200151X)


```python
print('Lengthscales for HL: \n')
print(ker1.rbf.lengthscale.values)
print('\n Lengthscales for CL: \n')
print(ker2.rbf.lengthscale.values)
```

    Lengthscales for HL: 
    
    [1.         1.         1.         1.         1.         5.92876918
     0.58579351 6.813832  ]
    
     Lengthscales for CL: 
    
    [ 1.          1.          1.          1.          1.         10.16274595
      0.72467334 11.32896314]
    

According to the lengthscales, X6 and X8, that correspond to orientation and glazing area distribution, are the most important attributes for predicting both, HL and CL. However, these lengthscales are even larger for CL, which means that they are even more important for predicting this variable. Then, X1, X2, X3, X4 and X5 (Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height) have the same weight, so they are equally important. Finally, X7, which is glazing area, is the one with least impact.

Surprinsingly, our results differ from those obtained in the paper. On the one hand, X7 is set as the most important feature, which is indeed our least important one. On the other hand, X6 and X8 are not considered particularly important. Also, X1 and X2 have a similar importance, but X4 and X5 don't. 

The dissimilarities between both results are significant and they lay on the particular model used for predicting. 




## Operations on the test set

#### Computing the test mean absolute error error and the test mean square error (MSE)  using the GP posterior mean and the optimized hyperparameters. Comparing the results with Tables 6 and 7 in this [paper](https://www.sciencedirect.com/science/article/pii/S037877881200151X).


```python
from sklearn import metrics

### HL
meanYtest,_ = m1.predict(X_test.values,full_cov=True)
MAE1 =  metrics.mean_absolute_error(y_test['Y1'], meanYtest)
MSE1 = metrics.mean_squared_error(y_test['Y1'], meanYtest)

### CL
meanYtest,_ = m2.predict(X_test.values,full_cov=True)
MAE2 =  metrics.mean_absolute_error(y_test['Y2'], meanYtest)
MSE2 = metrics.mean_squared_error(y_test['Y2'], meanYtest)

print('HL')
print('MAE: ',MAE1)
print('MSE: ', MSE1)
print('\nCL')
print('MAE: ',MAE2)
print('MSE: ', MSE2)
```

    HL
    MAE:  0.7939588932938333
    MSE:  1.0474830076200583
    
    CL
    MAE:  1.2465855727000994
    MSE:  3.3526051320668326
    

 In the paper two machine learning techniques have been applied: IRL and RF. 

When predicting HL, the obtained MAE and MSE are 0.51 and 1.03 for RF and 2.14 and 9.87 for IRL. We can see that our model performs similarly to RF, but better than IRL. However, the MAE using RF is slightly better than with GP. 

When predicting CL, the MAE and MSE obtained are 1.42 and 6.59 for RF and 2.21 and 11.46 for IRL. In this case, our model performs better than both, as the MAE and the MSE we have obtained is smaller.

#### b) Trying to improve the found results by using a more complicated kernel.

In order to inspect which kernel combinations could perform better, we create a function that combines three typical kernels (linear, periodic and RBF) in two possible ways: by adding them or by multiplying them. In total, there are six different combinations. Then, we the MAE and MSE of the model are computed given by each of the kernel combinations. 


```python
import itertools
covariance_functions = [GPy.kern.Linear(8), GPy.kern.StdPeriodic(8), GPy.kern.RBF(8)]
operations = {'+': lambda x, y: x + y, '*': lambda x, y: x * y}

def search_kernel(Y_train, Y_test):
  mae = {}
  mse = {}
  for j, base_kernels in enumerate(itertools.combinations(covariance_functions, 2)):
    for k, (op_name, op) in enumerate(operations.items()):
        kernel = op(base_kernels[0], base_kernels[1])
        m = GPy.models.GPRegression(X_train,Y_train, kernel)
        m.optimize(messages=True,max_f_eval = 1000)
        meanYtest,_ = m.predict(X_test.values,full_cov=True)
        MAE =  metrics.mean_absolute_error( Y_test, meanYtest)
        MSE = metrics.mean_squared_error(Y_test, meanYtest)
        mae['{} {} {}'.format(base_kernels[0].name, op_name, base_kernels[1].name)] = MAE
        mse['{} {} {}'.format(base_kernels[0].name, op_name, base_kernels[1].name)] = MSE

  return mae, mse
        
```


```python
mae1, mse1 = search_kernel(y_train[['Y1']],y_test[['Y1']])
mae2, mse2 = search_kernel(y_train[['Y2']],y_test[['Y2']])
```


    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…



    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…


     /usr/local/lib/python3.7/dist-packages/GPy/kern/src/standard_periodic.py:131: RuntimeWarning:overflow encountered in true_divide
     /usr/local/lib/python3.7/dist-packages/GPy/kern/src/standard_periodic.py:132: RuntimeWarning:invalid value encountered in sin
     /usr/local/lib/python3.7/dist-packages/GPy/kern/src/standard_periodic.py:148: RuntimeWarning:overflow encountered in true_divide
     /usr/local/lib/python3.7/dist-packages/GPy/kern/src/standard_periodic.py:150: RuntimeWarning:invalid value encountered in sin
     /usr/local/lib/python3.7/dist-packages/GPy/kern/src/standard_periodic.py:153: RuntimeWarning:invalid value encountered in cos
     /usr/local/lib/python3.7/dist-packages/GPy/kern/src/standard_periodic.py:153: RuntimeWarning:overflow encountered in true_divide
    


    HBox(children=(VBox(children=(IntProgress(value=0, max=1000), HTML(value=''))), Box(children=(HTML(value=''),)…


    HL
    MAE:  {'linear + std_periodic': 0.4423731764552061, 'linear * std_periodic': 0.3601268702853274, 'linear + rbf': 0.6427676955648612, 'linear * rbf': 0.5281868707873665, 'std_periodic + rbf': 0.4680476517833403, 'std_periodic * rbf': 0.5098376251039773}
    MSE:  {'linear + std_periodic': 0.4087353543728157, 'linear * std_periodic': 0.2767542992724571, 'linear + rbf': 0.7900318678788842, 'linear * rbf': 0.4671872181919311, 'std_periodic + rbf': 0.41111183450368277, 'std_periodic * rbf': 0.46722754573043984}
    
    CL
    MAE:  {'linear + std_periodic': 1.0774273828189675, 'linear * std_periodic': 1.2510771187114857, 'linear + rbf': 0.6948614756971624, 'linear * rbf': 0.8571198020247941, 'std_periodic + rbf': 0.8324040029972581, 'std_periodic * rbf': 0.7534990053784942}
    MSE:  {'linear + std_periodic': 2.404376935459638, 'linear * std_periodic': 3.137278607254643, 'linear + rbf': 0.8391915333090166, 'linear * rbf': 1.7623127698175312, 'std_periodic + rbf': 1.2564859673012663, 'std_periodic * rbf': 1.2969119721050146}
    

In the following tables the performance of these models with the baseline model trained will be computed..



*   HL







```python
results1=pd.DataFrame(data=mae1, index=[0])
results1.loc[1]=list(mse1.values())
results1['baseline']=[0.793958, 1.047483]
results1['metric']=['MAE', 'MSE']
results1.set_index('metric', inplace=True)
results1
```





  <div id="df-421d51ab-ebe3-401c-9a16-93f35fca4fc1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>linear + std_periodic</th>
      <th>linear * std_periodic</th>
      <th>linear + rbf</th>
      <th>linear * rbf</th>
      <th>std_periodic + rbf</th>
      <th>std_periodic * rbf</th>
      <th>baseline</th>
    </tr>
    <tr>
      <th>metric</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE</th>
      <td>0.442373</td>
      <td>0.360127</td>
      <td>0.642768</td>
      <td>0.528187</td>
      <td>0.468048</td>
      <td>0.509838</td>
      <td>0.793958</td>
    </tr>
    <tr>
      <th>MSE</th>
      <td>0.408735</td>
      <td>0.276754</td>
      <td>0.790032</td>
      <td>0.467187</td>
      <td>0.411112</td>
      <td>0.467228</td>
      <td>1.047483</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-421d51ab-ebe3-401c-9a16-93f35fca4fc1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-421d51ab-ebe3-401c-9a16-93f35fca4fc1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-421d51ab-ebe3-401c-9a16-93f35fca4fc1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




In this case, all the new models perform better than the previous one. Also, except linear+rbf and linear*rbf, all the methods tried are better than the MAE using RF(0.51) in the paper, and all the methods have less MSE than the MSE of using RF(1.03) in the paper. In particular, it could be seen that multiplying a linear kernel by a standard periodic kernel it could be reduced the baseline MAE and MSE in more than a 50% and 70% , respectively.



*   CL





```python
results2=pd.DataFrame(data=mae2, index=[0])
results2.loc[1]=list(mse2.values())
results2['baseline']=[1.246585, 3.352605]
results2['metric']=['MAE', 'MSE']
results2.set_index('metric', inplace=True)
results2
```





  <div id="df-9df24358-e771-4a62-add6-2885d8bb7be1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>linear + std_periodic</th>
      <th>linear * std_periodic</th>
      <th>linear + rbf</th>
      <th>linear * rbf</th>
      <th>std_periodic + rbf</th>
      <th>std_periodic * rbf</th>
      <th>baseline</th>
    </tr>
    <tr>
      <th>metric</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE</th>
      <td>1.077427</td>
      <td>1.251077</td>
      <td>0.694861</td>
      <td>0.857120</td>
      <td>0.832404</td>
      <td>0.753499</td>
      <td>1.246585</td>
    </tr>
    <tr>
      <th>MSE</th>
      <td>2.404377</td>
      <td>3.137279</td>
      <td>0.839192</td>
      <td>1.762313</td>
      <td>1.256486</td>
      <td>1.296912</td>
      <td>3.352605</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9df24358-e771-4a62-add6-2885d8bb7be1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9df24358-e771-4a62-add6-2885d8bb7be1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9df24358-e771-4a62-add6-2885d8bb7be1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




In this case, the kernel that achieves the lowest error is the addition of a linear kernel plus a RBF. The model with this kernel reduces the baseline MAE and MSE in more than a 40% and 75%, respectively. 

Surprisingly, it could be found out that the model that best performs predicting HL is the one that performs the worst for CL with respect to the new models, having a very similar performance to the baseline model.   
By contrast, the kernel that achieves the best performance when predicting CL is the one that performs the worst predicting HL. However, this kernel still results in a significant improvement with respect to the baseline. 
