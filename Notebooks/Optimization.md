---
layout: archive-dates
title: Implementation of some machine learning optimization methods
toc: true
---

## Project Description:

In this project, some optimization method will be studied such as Gradient, newton, quasi-newton etc. First the random data will be generated from a predefined linear regression model where $5\leq \beta_i \leq5$ and $i = 0,...,K$ with K = 120 independent variables $X = (X_1,X_2,...,X_K)$ and n = 1000 observations.
In the project, it is seeked to adjust a multiple linear regression model to explain variable Y as a function of the other variables X, i.e.,($ Y = \beta 'X + \epsilon $)  by using "Ridge Regression":

$$ \min_{\beta} ||y-X\beta||^2_2 + \rho ||\beta||^2_2$$

where $\rho$ is a parameter and consider it fixed to a given value $\rho = 1$



```python
# libraries
%matplotlib notebook
import numpy as np
from time import time
np.random.seed(100459259)

# number of preditors and number of observations
nvars = 120 ## K value
nsample = 1000
# beta between -5 and 5
beta = np.random.randint(-5,5,size=([nvars+1,1]))
## X model matrix
X0 = np.ones([nsample,1]) # the first column has all values equal to one for the coefficients of beta0
X1 = np.random.uniform(0,10,([nsample,nvars]))
X = np.concatenate([X0, X1],axis=1)
## Values for the normal errors
error = np.random.normal(0,1,(nsample,1))
## Values for the y's
Y = np.dot(X,beta) + error
```

### Visualizing some variables of the random data


```python
import matplotlib.pyplot as plt
%matplotlib inline

Y_p = np.array(Y)

# Plot four of the variables
plt.figure(figsize=(5, 5))
plt.subplot(2, 2, 1)
plt.scatter(X[:,1], Y_p),plt.xlabel('X1'),plt.ylabel('Y_p')

plt.subplot(2, 2, 2)
plt.scatter(X[:,2], Y_p),plt.xlabel('X2'),plt.ylabel('Y_p')

plt.subplot(2, 2, 3)
plt.scatter(X[:,3], Y_p),plt.xlabel('X3'),plt.ylabel('Y_p')

plt.subplot(2, 2, 4)
plt.scatter(X[:,4], Y_p),plt.xlabel('X4'),plt.ylabel('Y_p')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace = 0.4, hspace = 0.4)
plt.show()
```


    
![png](output_3_0.png)
    


## a) Estimating the value of the regression coefficients by implementing the analytical solution
For ridge regression, there exists an analytical solution. \
The exact solution is: $\beta^*=(X^T X + \rho I)^{-1}X^T y$



```python
from numpy.linalg import inv
from time import time

rho=1 # it represent the p value in the formula. I am assuming that it is 1.
time_start = time()
beta_ridge_exact=np.dot(np.dot(inv(np.dot((X.T),X)+rho*np.identity(nvars+1)),X.T),Y) ## I= identity matrix
time_end = time()

time_elapsed = (time_end - time_start)
# Print the results
print('Values of the (exact) ridge regression coefficients:')
for i in range(nvars+1):
    print('beta %3d %7.3f' %(i,beta_ridge_exact[i]))
print('Elapsed time = %8.5f' %(time_elapsed))
```

    Values of the (exact) ridge regression coefficients:
    beta   0  -0.988
    beta   1   1.001
    beta   2  -3.979
    beta   3   0.987
    beta   4  -2.994
    beta   5   3.000
    beta   6   1.983
    beta   7   2.011
    beta   8   2.995
    beta   9  -5.005
    beta  10   0.994
    beta  11  -0.999
    beta  12  -1.010
    beta  13   0.008
    beta  14   1.974
    beta  15   1.990
    beta  16   3.998
    beta  17  -1.983
    beta  18   3.982
    beta  19  -3.999
    beta  20  -3.008
    beta  21   1.997
    beta  22   0.980
    beta  23   2.016
    beta  24  -0.018
    beta  25  -2.013
    beta  26  -0.997
    beta  27  -2.977
    beta  28  -0.973
    beta  29   2.999
    beta  30  -4.007
    beta  31   2.018
    beta  32  -3.028
    beta  33  -1.970
    beta  34  -1.011
    beta  35  -3.009
    beta  36   0.014
    beta  37  -4.980
    beta  38   4.010
    beta  39  -4.997
    beta  40  -4.016
    beta  41  -0.992
    beta  42   2.005
    beta  43  -5.003
    beta  44  -3.995
    beta  45   3.996
    beta  46   2.007
    beta  47   0.993
    beta  48  -2.004
    beta  49   0.022
    beta  50  -1.009
    beta  51  -4.020
    beta  52   0.009
    beta  53   3.008
    beta  54  -4.002
    beta  55  -3.007
    beta  56  -3.996
    beta  57  -2.007
    beta  58  -4.007
    beta  59  -3.988
    beta  60  -0.988
    beta  61  -0.011
    beta  62  -4.003
    beta  63  -2.005
    beta  64  -2.994
    beta  65  -1.980
    beta  66   3.016
    beta  67  -4.987
    beta  68   3.980
    beta  69  -2.007
    beta  70  -0.975
    beta  71  -2.010
    beta  72   3.010
    beta  73  -3.000
    beta  74   1.997
    beta  75  -4.991
    beta  76  -0.002
    beta  77  -0.003
    beta  78  -1.995
    beta  79   2.013
    beta  80   0.988
    beta  81  -0.006
    beta  82   2.032
    beta  83  -0.016
    beta  84  -2.018
    beta  85   3.015
    beta  86  -4.990
    beta  87  -3.007
    beta  88   1.005
    beta  89  -0.994
    beta  90   2.003
    beta  91  -0.964
    beta  92   4.001
    beta  93  -1.000
    beta  94  -4.000
    beta  95   3.006
    beta  96   1.010
    beta  97  -0.017
    beta  98   1.986
    beta  99  -1.995
    beta 100   2.988
    beta 101   1.000
    beta 102   3.007
    beta 103   3.991
    beta 104  -5.009
    beta 105   1.999
    beta 106   2.012
    beta 107  -0.990
    beta 108   4.017
    beta 109  -1.013
    beta 110   3.981
    beta 111   0.986
    beta 112  -0.031
    beta 113  -2.013
    beta 114  -5.008
    beta 115  -3.991
    beta 116   0.001
    beta 117   1.995
    beta 118   2.009
    beta 119   1.988
    beta 120  -4.009
    Elapsed time =  0.03599
    

## b) Estimating the value of the regression coefficients by using the function minimize from the Python module Scipy.optimize


### Defining ridge regression, gradient and hess of ridge regression formulas and creating functions
In order to try different solvers for the minimization problem, first ridge regression, gradient of ridge and hess of ridge functions should be defined:

As the ridge regression open formula is 
$$ \min_{\beta} (Y-\beta^TX)^T(Y-\beta^TX) + \rho\beta^T\beta$$ 
Then to find the gradient of the ridge regression, The derivation of the formula is:
$$ −2X^T(Y−\beta^TX) + 2\rho\beta $$
And, the hess of the ridge regression will be:
$$ 2X^TX + 2\rho$$


```python
## I am assuming that p(rho) is 1.
# ridge regression function
def ridge_reg(beta_rd,X,Y):
    beta_rd = np.matrix(beta_rd)
    z = Y - np.dot(X,beta_rd.T)
    val_f = np.dot(z.T,z) +  1*np.dot(beta_rd,beta_rd.T)
    val_ff = np.squeeze(np.asarray(val_f))
    return val_ff # OF 

# gradient function
def ridge_reg_der(beta_rd,X,Y):
    beta_rd = np.matrix(beta_rd)
    gradient = -2*np.dot((Y-np.dot(X,(beta_rd).T)).T,X) + 2 * 1 * beta_rd
    aa = np.squeeze(np.asarray(gradient))
    return aa

# hession function
def ridge_reg_hess(beta_rd,X,Y):
    hess_value = 2*np.dot(X.T,X) + 2*1
    return hess_value
```

### Trying different methods for minimizing ridge problem

In this section, six methods will be tried to minimize the ridge problem. 
First, all methods are written into a list called methods, and then with for loop, each method has been tried, respectively. As each method has different type of paramaters in minimize function such as jac or hess, if loop has been created to restrict this situation. \
As a result, the methods which had the minimum error and minimum objective function have been printed.


```python
# Definition of the optimization problem
from scipy.optimize import minimize
from time import time

## creating an empty list for each method's functions.
OF_list = list()
error_list = list()

## trying 4 different methods
methods = ['Nelder-Mead','COBYLA','Newton-CG','BFGS','Powell','CG']
for m in methods:
    if (m == 'Newton-CG'):
        hess=ridge_reg_hess
        jac=ridge_reg_der
    elif (m == 'BFGS' or m=='CG'):
        hess=None
        jac=ridge_reg_der
    else:
        jac=None
        hess=None
    print('\n-{} Method - Values obtained for the Ridge problem:'.format(m))
    
    time_start = time()
    res = minimize(ridge_reg, np.zeros(nvars+1), args=(X, Y), jac=jac, hess=hess, method=m, options={'maxiter': 10000,'disp': True})
    time_end = time()

    print('Elapsed time = %8.5f' %(time_end-time_start))
    OF_list.append([m,res.fun])
    print('Number of Function: {}'.format(res.fun))

    if m != 'COBYLA': print('Number of Iterations: {}'.format(res.nit))

    ##error values
    err_val = np.linalg.norm(beta_ridge_exact.T-res.x,ord=2)/np.linalg.norm(beta_ridge_exact.T,ord=2)
    error_list.append([m,err_val])
    print('Error in values of coefficients = %8.4f' %err_val)

## printing the estimated betas with the lowest OF method:
print('minimum error value found= ',min(error_list, key=lambda x: x[1]))
print('minimum current function value to be minimized= ',min(OF_list, key=lambda x: x[1]))
```

    
    -Nelder-Mead Method - Values obtained for the Ridge problem:
    Warning: Maximum number of iterations has been exceeded.
    Elapsed time =  2.72899
    Number of Function: 81116065.31757173
    Number of Iterations: 10000
    Error in values of coefficients =   0.9970
    
    -COBYLA Method - Values obtained for the Ridge problem:
    Elapsed time = 48.48046
    Number of Function: 197669.9303219118
    Error in values of coefficients =   0.2092
    
    -Newton-CG Method - Values obtained for the Ridge problem:
    Optimization terminated successfully.
             Current function value: 1841.749069
             Iterations: 16
             Function evaluations: 21
             Gradient evaluations: 21
             Hessian evaluations: 16
    Elapsed time =  0.03703
    Number of Function: 1841.7490692331621
    Number of Iterations: 16
    Error in values of coefficients =   0.0010
    
    -BFGS Method - Values obtained for the Ridge problem:
    Optimization terminated successfully.
             Current function value: 1841.745770
             Iterations: 141
             Function evaluations: 256
             Gradient evaluations: 256
    Elapsed time =  0.16946
    Number of Function: 1841.745770005918
    Number of Iterations: 141
    Error in values of coefficients =   0.0000
    
    -Powell Method - Values obtained for the Ridge problem:
    Optimization terminated successfully.
             Current function value: 291359.147775
             Iterations: 1385
             Function evaluations: 1510300
    Elapsed time = 188.01013
    Number of Function: 291359.1477746054
    Number of Iterations: 1385
    Error in values of coefficients =   9.0136
    
    -CG Method - Values obtained for the Ridge problem:
    Warning: Desired error not necessarily achieved due to precision loss.
             Current function value: 1841.745770
             Iterations: 184
             Function evaluations: 469
             Gradient evaluations: 458
    Elapsed time =  0.15290
    Number of Function: 1841.7457701092994
    Number of Iterations: 184
    Error in values of coefficients =   0.0000
    minimum error value found=  ['BFGS', 2.2148306613552646e-12]
    minimum current function value to be minimized=  ['BFGS', 1841.745770005918]
    

### Comparison of each method performance 

- **Number of iterations:** Nelder-Mead Method did not converge, which means that the maximum number of iterations (10000) has been exceeded. Also Nelder-Mead Method has the highest error rate. Netwon-CG method used the minimum number of iterations (16) to achive the optimal solution, and BFGS method followed the second minimum number of iterations (141), and CG method has the third minimum(184). Those three methods' error rates are the minimum rates among other methods as well.

- **Number of function:** The minimum objective function has been achieved with BFGS method, which is almost the same with CG and Newton-CG methods. The other methods' functions have been calculated quite high, accordingly, these methods' error rates had been observed really high.

- **Gradient evaluations:** Gradient method is used by Newton-CG, CG and BFGS methods. The least number of gradient evaluation has been observed at 21 with Newton-CG method. Then in BFGS, number of evaluations for gradient has been observed at 256. Overall, CG method had the highest evaluations with 458.

- **Hessian evaluations:** Hessian method is used by only Newton-CG method and number of evaluations has been observed at 16, which is quite low. 
- **Execution time:** As the Newton-CG method had the least iterations and evaluations values, which means the method could converge early, the execution time is the least one among others. Moreover, BFGS and CG methods had few execution times. On the other hand, Nelder-Mead,COBYLA,and Powell methods take much longer.

### Values of the ridge regression coefficients with the BFGS method that had minimum error

After trying different methods, BFGS method which is one of the quasi-newton methods had the least error rate, therefore, betas that estimated with this method are displaying in this section.


```python
## winning method: BFGS(quasi-newton method)
res = minimize(ridge_reg, np.zeros(nvars+1), args=(X, Y), jac=ridge_reg_der, hess=None, method='BFGS', options={'maxiter': 10000,'disp': True})
print('\nValues of the least squares coefficients obtained with Nelder-Mead:')
for i in range(nvars+1):
    print('beta %3d %7.3f' %(i,res.x[i]))
```

    Optimization terminated successfully.
             Current function value: 1841.745770
             Iterations: 141
             Function evaluations: 256
             Gradient evaluations: 256
    
    Values of the least squares coefficients obtained with Nelder-Mead:
    beta   0  -0.988
    beta   1   1.001
    beta   2  -3.979
    beta   3   0.987
    beta   4  -2.994
    beta   5   3.000
    beta   6   1.983
    beta   7   2.011
    beta   8   2.995
    beta   9  -5.005
    beta  10   0.994
    beta  11  -0.999
    beta  12  -1.010
    beta  13   0.008
    beta  14   1.974
    beta  15   1.990
    beta  16   3.998
    beta  17  -1.983
    beta  18   3.982
    beta  19  -3.999
    beta  20  -3.008
    beta  21   1.997
    beta  22   0.980
    beta  23   2.016
    beta  24  -0.018
    beta  25  -2.013
    beta  26  -0.997
    beta  27  -2.977
    beta  28  -0.973
    beta  29   2.999
    beta  30  -4.007
    beta  31   2.018
    beta  32  -3.028
    beta  33  -1.970
    beta  34  -1.011
    beta  35  -3.009
    beta  36   0.014
    beta  37  -4.980
    beta  38   4.010
    beta  39  -4.997
    beta  40  -4.016
    beta  41  -0.992
    beta  42   2.005
    beta  43  -5.003
    beta  44  -3.995
    beta  45   3.996
    beta  46   2.007
    beta  47   0.993
    beta  48  -2.004
    beta  49   0.022
    beta  50  -1.009
    beta  51  -4.020
    beta  52   0.009
    beta  53   3.008
    beta  54  -4.002
    beta  55  -3.007
    beta  56  -3.996
    beta  57  -2.007
    beta  58  -4.007
    beta  59  -3.988
    beta  60  -0.988
    beta  61  -0.011
    beta  62  -4.003
    beta  63  -2.005
    beta  64  -2.994
    beta  65  -1.980
    beta  66   3.016
    beta  67  -4.987
    beta  68   3.980
    beta  69  -2.007
    beta  70  -0.975
    beta  71  -2.010
    beta  72   3.010
    beta  73  -3.000
    beta  74   1.997
    beta  75  -4.991
    beta  76  -0.002
    beta  77  -0.003
    beta  78  -1.995
    beta  79   2.013
    beta  80   0.988
    beta  81  -0.006
    beta  82   2.032
    beta  83  -0.016
    beta  84  -2.018
    beta  85   3.015
    beta  86  -4.990
    beta  87  -3.007
    beta  88   1.005
    beta  89  -0.994
    beta  90   2.003
    beta  91  -0.964
    beta  92   4.001
    beta  93  -1.000
    beta  94  -4.000
    beta  95   3.006
    beta  96   1.010
    beta  97  -0.017
    beta  98   1.986
    beta  99  -1.995
    beta 100   2.988
    beta 101   1.000
    beta 102   3.007
    beta 103   3.991
    beta 104  -5.009
    beta 105   1.999
    beta 106   2.012
    beta 107  -0.990
    beta 108   4.017
    beta 109  -1.013
    beta 110   3.981
    beta 111   0.986
    beta 112  -0.031
    beta 113  -2.013
    beta 114  -5.008
    beta 115  -3.991
    beta 116   0.001
    beta 117   1.995
    beta 118   2.009
    beta 119   1.988
    beta 120  -4.009
    

## c) Estimating the value of the regression coefficients by implementing:
### **1. Gradient Method**

#### Gradient Method Description and Formula
In this method, from an initial iterate $x_0$ descent directions which represents $p_k=-\nabla f(x_k)$ are computed.
Movement of the gradient is following:
$$x_{k+1} = x_k + \alpha_k\ p_k$$
Number of k iterations take place until the algortihm converges to a local solution

- A precision variable is set in the algorithm that calculates the difference between two consecutive "x" values. If the difference between x values in 2 consecutive iterations is less than the specified precision, the algorithm stops.


```python
## Initial values for each parameter:
iterations = 30000 # maximum number of iterations
OF_iteration = list()
tolerance_iteration = list()

beta = np.zeros(nvars+1) # initial value for beta
alpha = 0.0000001   
precision = 0.00001 #This tells us when to stop the algorithm
previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y))
iter_count = 0
## process starts
time_start = time()
while (iter_count < iterations) and (previous_step_size > precision):

    beta = beta - alpha*ridge_reg_der(beta,X,Y) ## ridge_reg_der is gradient function and minus indicates gradient descent direction
 
    OF_iteration.append(ridge_reg(beta, X, Y)) ## calculating the OF (cost) of ridge regression with ridge_reg function
    tolerance_iteration.append(previous_step_size)
    
    iter_count = iter_count + 1 ## increasing iteration number
    previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2) # tolerance: #the changes between x_k+1 and x_k

## process ends
time_end = time() 

## results
if iter_count==iterations: print('\nGradient descent does not converge.')
print('\nValues of the ridge regression coefficients - gradient method:')
print('Elapsed time = %8.5f' %(time_end- time_start))
print('\nNumber of iterations = %5.0f' %iter_count)
print('Objective function   = %11.5f' %OF_iteration[-1]) # the last, minimized OF.
print('Optimality tolerance = %11.5f' %previous_step_size)
beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-beta,ord=2)/np.linalg.norm(beta,ord=2)
print('\nBeta coefficient error = %10.5f' %beta_error)

## estimated betas
print('beta %-9s %7.3f' %('intercept',beta[0]))
col_list = ['X' + str(x) for x in range(0,nvars+1)]
for i in np.arange(1,nvars+1):
    print('beta %-9s %7.3f' %(col_list[i],beta[i]))
```

    
    Gradient descent does not converge.
    
    Values of the ridge regression coefficients - gradient method:
    Elapsed time =  7.56488
    
    Number of iterations = 30000
    Objective function   =  1844.49271
    Optimality tolerance =     6.26020
    
    Beta coefficient error =    0.02900
    beta intercept  -0.110
    beta X1          0.999
    beta X2         -3.980
    beta X3          0.986
    beta X4         -2.996
    beta X5          2.999
    beta X6          1.981
    beta X7          2.011
    beta X8          2.995
    beta X9         -5.006
    beta X10         0.992
    beta X11        -1.000
    beta X12        -1.012
    beta X13         0.008
    beta X14         1.972
    beta X15         1.989
    beta X16         3.996
    beta X17        -1.984
    beta X18         3.980
    beta X19        -4.001
    beta X20        -3.010
    beta X21         1.995
    beta X22         0.978
    beta X23         2.015
    beta X24        -0.019
    beta X25        -2.013
    beta X26        -0.998
    beta X27        -2.979
    beta X28        -0.975
    beta X29         2.998
    beta X30        -4.009
    beta X31         2.016
    beta X32        -3.028
    beta X33        -1.972
    beta X34        -1.013
    beta X35        -3.010
    beta X36         0.012
    beta X37        -4.981
    beta X38         4.008
    beta X39        -4.998
    beta X40        -4.018
    beta X41        -0.993
    beta X42         2.003
    beta X43        -5.003
    beta X44        -3.997
    beta X45         3.995
    beta X46         2.005
    beta X47         0.991
    beta X48        -2.005
    beta X49         0.020
    beta X50        -1.010
    beta X51        -4.022
    beta X52         0.007
    beta X53         3.006
    beta X54        -4.003
    beta X55        -3.008
    beta X56        -3.997
    beta X57        -2.010
    beta X58        -4.008
    beta X59        -3.989
    beta X60        -0.990
    beta X61        -0.011
    beta X62        -4.004
    beta X63        -2.006
    beta X64        -2.996
    beta X65        -1.981
    beta X66         3.014
    beta X67        -4.989
    beta X68         3.978
    beta X69        -2.008
    beta X70        -0.977
    beta X71        -2.011
    beta X72         3.008
    beta X73        -3.003
    beta X74         1.996
    beta X75        -4.992
    beta X76        -0.004
    beta X77        -0.004
    beta X78        -1.997
    beta X79         2.012
    beta X80         0.987
    beta X81        -0.007
    beta X82         2.031
    beta X83        -0.017
    beta X84        -2.019
    beta X85         3.014
    beta X86        -4.992
    beta X87        -3.008
    beta X88         1.003
    beta X89        -0.996
    beta X90         2.002
    beta X91        -0.966
    beta X92         3.999
    beta X93        -1.002
    beta X94        -4.002
    beta X95         3.003
    beta X96         1.008
    beta X97        -0.018
    beta X98         1.985
    beta X99        -1.997
    beta X100        2.986
    beta X101        0.998
    beta X102        3.006
    beta X103        3.990
    beta X104       -5.010
    beta X105        1.999
    beta X106        2.011
    beta X107       -0.991
    beta X108        4.016
    beta X109       -1.014
    beta X110        3.980
    beta X111        0.984
    beta X112       -0.034
    beta X113       -2.013
    beta X114       -5.009
    beta X115       -3.993
    beta X116        0.001
    beta X117        1.994
    beta X118        2.008
    beta X119        1.987
    beta X120       -4.010
    

- As seen in the above results, the gradient descent method did not converge with 30000 iterations. Optimality tolerance has been calculated 6.26, which is quite large than precision value (0.00001). However, the error rate of the method is quite small (0.02900).

#### Plotting Objective Function and Tolerance evaluations through iterations

As seen in the visuals, OF and tolerance values tend to decrease through iterations. For OF, the function decreases sharply at the beginning. For tolerance values, there is a sharp decrease until the algorithm reaches to 15000 iterations. Then the tolerance values are stabil.


```python
plt.plot(OF_iteration[1:iter_count-1]),plt.ylabel('Objective function'),plt.xlabel('Iterations')
plt.show()

plt.plot(np.log(tolerance_iteration[1:iter_count-1])),plt.ylabel('Log opt error'),plt.xlabel('Iterations')
plt.show()
```


    
![png](output_19_0.png)
    



    
![png](output_19_1.png)
    


#### **Implementing Armijo Rule to adjust alpha for Gradient Descent**
|        |
|  ----  |
|![image.png](attachment:bb783dab-d554-4fa2-8882-5a4a42f8efd4.png)|

The ArmijoLineSearch function has been written according to the above formula's left hand and right hand sides. When this equation can NOT be obtained, the function reaches the optimal alpha value and the while loop stops.


```python
def ArmijoLineSearch(beta, alpha, sigma, beta_decrease):
   
    while not ridge_reg((beta+alpha*descent_direction),X,Y) <= (ridge_reg(beta,X,Y) + sigma*alpha*np.dot(descent_direction,ridge_reg_der(beta,X,Y))):
        alpha = alpha*beta_decrease
    
    return alpha
```


```python
## empty lists to append inside each iteration value
OF_iteration = list()
tolerance_iteration = list()
alpha_iteration = list()

## Initial values for the variables
iterations = 30000 # maximum number of iterations
alpha = 1 ## initial  
alpha_iteration.append(alpha)
precision = 0.00001 #This tells us when to stop the algorithm

beta = np.zeros(nvars+1) # xk # initial value for beta 
previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2)

iter_count = 0
## process starts
time_start = time()
while (iter_count < iterations) and (previous_step_size > precision):
    descent_direction = -ridge_reg_der(beta,X,Y)
    #calculate new beta,f(x), ve gradient func

    alpha = ArmijoLineSearch(beta, alpha=alpha, sigma=0.1, beta_decrease=0.1)
    
    beta = beta + alpha*descent_direction
    
    previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2)
    iter_count = iter_count + 1 ## increasing iteration number
    
    ## appending to lists
    alpha_iteration.append(alpha)
    OF_iteration.append(ridge_reg(beta, X, Y))
    tolerance_iteration.append(previous_step_size)
        
## process ends
time_end = time() 
## results
if iter_count==iterations: print('\nGradient descent does not converge.')
print('\nValues of the ridge regression coefficients - gradient method:')
print('Elapsed time = %8.5f' %(time_end- time_start))
print('\nNumber of iterations = %5.0f' %iter_count)
print('\nNumber of alpha = ' ,alpha_iteration[-1])
print('Objective function   = %11.5f' %OF_iteration[-1]) # the last, minimized OF.
print('Optimality tolerance = %11.5f' %previous_step_size)
beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-beta,ord=2)/np.linalg.norm(beta,ord=2)
print('\nBeta coefficient error = %10.5f' %beta_error)
```

    
    Gradient descent does not converge.
    
    Values of the ridge regression coefficients - gradient method:
    Elapsed time = 14.40957
    
    Number of iterations = 30000
    
    Number of alpha =  1.0000000000000005e-07
    Objective function   =  1844.49271
    Optimality tolerance =     6.26020
    
    Beta coefficient error =    0.02900
    

- As seen in the above results, even though, the alpha is smaller than the initial value, the algortihm does not converge with 30000 iterations.

##### Plotting Alpha evaluations through iterations


```python
plt.plot(alpha_iteration),plt.ylabel('Alpha'),plt.xlabel('Iterations')
plt.ticklabel_format(style='plain')    # to prevent scientific notation.
plt.show()
```


    
![png](output_25_0.png)
    


### 2. Newton's Method
#### Newton's Method Description and Formula 

In this method, again, from an initial iterate $x_0$, descent direction which is $p_k=-(\nabla^2 f(x_k))^{-1} \nabla f(x_k)$ are computed whenever $\nabla^2 f(x_k)$ is nonsingular. In each iteration k, the movement of the method is following until it convergences to a local solution.
$$x_{k+1} = x_k + \alpha_k\ p_k$$



```python
## empty lists to append inside each iteration value
OF_iteration = list()
tolerance_iteration = list()
alpha_iteration = list()

## Initial values for the variables
iterations = 30000 # maximum number of iterations
alpha = 0.01  ## initial  
precision = 0.000001 #This tells us when to stop the algorithm

beta = np.zeros(nvars+1) # xk # initial value for beta 
previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y))
    
iter_count = 0
## process starts
time_start = time()
while (iter_count < iterations) and (previous_step_size > precision):
    
    gradient = ridge_reg_der(beta,X,Y)
    hess = ridge_reg_hess(beta,X,Y)
    descent_direction = -np.dot(np.linalg.inv(hess),gradient) # Descent direction
    
    #calculate new beta,f(x), ve gradient func
    #alpha = ArmijoLineSearch(beta, alpha=alpha, sigma=0.01, beta_decrease=0.1)
    
    beta = beta + alpha*descent_direction
    
    previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2)
    iter_count = iter_count + 1 ## increasing iteration number
    
    ## appending to lists
    alpha_iteration.append(alpha)
    OF_iteration.append(ridge_reg(beta, X, Y))
    tolerance_iteration.append(previous_step_size)
        
## process ends
time_end = time() 
## results
if iter_count==iterations: 
    print("\nNewton's method does not converge.")
else: 
    print("\nNewton's method does converge.")
print('\nValues of the ridge regression coefficients - gradient method:')
print('Elapsed time = %8.5f' %(time_end- time_start))
print('\nNumber of iterations = %5.0f' %iter_count)
print('\nNumber of alpha = ' ,alpha_iteration[iter_count-1])
print('Objective function   = %11.5f' %OF_iteration[iter_count-1]) # the last, minimized OF.
print('Optimality tolerance = %11.5f' %previous_step_size)
beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-beta,ord=2)/np.linalg.norm(beta,ord=2)
print('\nBeta coefficient error = %10.5f' %beta_error)
```

    
    Newton's method does converge.
    
    Values of the ridge regression coefficients - gradient method:
    Elapsed time =  3.12055
    
    Number of iterations =  3092
    
    Number of alpha =  0.01
    Objective function   =  1841.74577
    Optimality tolerance =     0.00000
    
    Beta coefficient error =    0.00000
    

- As seen in the above results, Newton method does converge with 0.000001 precision value, 3092 iterations and 0.01 alpha value. The method reaches to almost zero error rate which is very successful.

#### Plotting Objective Function and Tolerance evaluations through iterations

As seen from graphs that OF has decreased sharply at the beginnning and the tolerance of each iteration has decreased continuesly until the end of iterations.


```python
# Plot results showing the evolution of the algorithm
plt.figure()
plt.plot(OF_iteration),plt.ylabel('Objective function'),plt.xlabel('Iterations')
plt.ticklabel_format(style='plain')    # to prevent scientific notation.
plt.show()

plt.plot(np.log(tolerance_iteration)),plt.ylabel('Log opt error'),plt.xlabel('Iterations')
plt.show()
```


    
![png](output_30_0.png)
    



    
![png](output_30_1.png)
    


#### Implementing Armijo Rule to adjust alpha for Newton method


```python
## empty lists to append inside each iteration value
OF_iteration = list()
tolerance_iteration = list()
alpha_iteration = list()

## Initial values for the variables
iterations = 10000 # maximum number of iterations
alpha = 1  ## initial  
precision = 0.0001 #This tells us when to stop the algorithm

beta = np.zeros(nvars+1) # xk # initial value for beta 
previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y))
    
iter_count = 0
## process starts
time_start = time()
while (iter_count < iterations) and (previous_step_size > precision):
    
    gradient = ridge_reg_der(beta,X,Y)
    hess = ridge_reg_hess(beta,X,Y)
    descent_direction = -np.dot(np.linalg.inv(hess),gradient) # Descent direction
    
    #calculate new beta,f(x), ve gradient func
    alpha = ArmijoLineSearch(beta, alpha=alpha, sigma=0.01, beta_decrease=0.1)
    
    beta = beta + alpha*descent_direction
    
    previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2)
    iter_count = iter_count + 1 ## increasing iteration number
    
    ## appending to lists
    alpha_iteration.append(alpha)
    OF_iteration.append(ridge_reg(beta, X, Y))
    tolerance_iteration.append(previous_step_size)
        
## process ends
time_end = time() 
## results
if iter_count==iterations: 
    print("\nNewton's method does not converge.")
else: 
    print("\nNewton's method does converge.")
print('\nValues of the ridge regression coefficients - gradient method:')
print('Elapsed time = %8.5f' %(time_end- time_start))
print('\nNumber of iterations = %5.0f' %iter_count)
print('\nNumber of alpha = ' ,alpha_iteration[-1])
print('Objective function   = %11.5f' %OF_iteration[-1]) # the last, minimized OF.
print('Optimality tolerance = %11.5f' %previous_step_size)
beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-beta,ord=2)/np.linalg.norm(beta,ord=2)
print('\nBeta coefficient error = %10.5f' %beta_error)
```

    
    Newton's method does converge.
    
    Values of the ridge regression coefficients - gradient method:
    Elapsed time =  0.01787
    
    Number of iterations =     9
    
    Number of alpha =  1
    Objective function   =  1841.74577
    Optimality tolerance =     0.00006
    
    Beta coefficient error =    0.00000
    

- From the above results, it can be interpretted that even though the armijo rule function has been started with 1 alpha value, the algorithm reached to optimality after only 9 iterations with the same alpha value. The error rate has been calculated very close to zero. 


```python
plt.plot(alpha_iteration),plt.ylabel('Alpha'),plt.xlabel('Iterations')
plt.ticklabel_format(style='plain')    # to prevent scientific notation.
plt.show()
```


    
![png](output_34_0.png)
    


### 3. Quasi-Newton method
#### Quasi-Newton method Description and Formula
|Quasi-Newton Formula|
| --- |
|![image.png](attachment:42ae299f-de84-4590-ac00-a14740b84efd.png)|

In this step, BFGS update rule has been examined.


```python
## empty lists to append inside each iteration value
OF_iteration = list()
tolerance_iteration = list()
alpha_iteration = list()

## Initial values for the variables
iterations = 10000 # maximum number of iterations
alpha = 0.1  ## initial  
precision = 0.0001 #This tells us when to stop the algorithm

## Initial parameters
beta = np.zeros(nvars+1) # xk # initial value for beta 
gradient = ridge_reg_der(beta,X,Y)
Bk = ridge_reg_hess(beta,X,Y) ## initial approximation to hessian matrix.
previous_step_size = np.linalg.norm(gradient)

iter_count = 0
## process starts
time_start = time()

while (iter_count < iterations) and (previous_step_size > precision):
    
    descent_direction = -np.dot(np.linalg.inv(Bk),gradient)
    
    #Compute the step length lambda with a line search procedure that satisfies Wolfe conditions. 
    alpha = ArmijoLineSearch(beta, alpha=alpha, sigma=0.01, beta_decrease=0.1)

    beta_1 = beta + alpha*descent_direction
    
    sk = beta_1-beta
    yk = ridge_reg_der(beta_1,X,Y) - ridge_reg_der(beta,X,Y)
    
    beta = beta_1

    eq1 = -np.dot(np.dot(Bk, sk),np.dot(Bk, sk).T)
    eq_denom1 = np.dot(sk.T, Bk, sk)
    divide1 = np.divide(eq1,eq_denom1, where=eq_denom1!=0) ## denominator cannot be zero.
    
    eq2 = np.dot(yk , yk.T)
    eq_denom2 = np.dot(yk.T , sk)
    divide2 = np.divide(eq2,eq_denom2, where=eq_denom2!=0) ## denominator cannot be zero.
    Bk = Bk + divide1 + divide2
    
    ##iterate 
    previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2)
    iter_count = iter_count + 1 ## increasing iteration number
    
    ## appending to lists
    OF_iteration.append(ridge_reg(beta, X, Y))
    alpha_iteration.append(alpha)
    tolerance_iteration.append(previous_step_size)
    
## process ends
time_end = time() 
## results
print('\nValues of the ridge regression coefficients - quasi-newton method:')
print('Elapsed time = %8.5f' %(time_end- time_start))
print('\nNumber of iterations = %5.0f' %iter_count)
print('\nNumber of alpha = ' ,alpha_iteration[-1])
print('Objective function   = %11.5f' %OF_iteration[-1]) # the last, minimized OF.
print('Optimality tolerance = %11.5f' %previous_step_size)
beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-beta,ord=2)/np.linalg.norm(beta,ord=2)
print('\nBeta coefficient error = %10.5f' %beta_error)
```

    
    Values of the ridge regression coefficients - quasi-newton method:
    Elapsed time = 15.21355
    
    Number of iterations = 10000
    
    Number of alpha =  1.000000000000001e-17
    Objective function   =  2300.58365
    Optimality tolerance =   522.20124
    
    Beta coefficient error =    0.35416
    


```python
# Plot results showing the evolution of the algorithm
plt.plot(alpha_iteration),plt.ylabel('Alpha'),plt.xlabel('Iterations')
plt.show()

```


    
![png](output_37_0.png)
    


## d) Estimating the value of the regression coefficients by implementing:
### **1.Coordinate gradient method**
| Coordinated Descent (CD) Formula |
| ---- |
|![image.png](attachment:d03ca2c5-c8e0-41f0-8fba-3c157a6dd658.png)|

In this step, the coordinated descent function has been created by using the above formula. 


```python
## Initial values for the variables and data containers
iterations = 4000 # maximum number of iterations
OF_iteration = list()

## initializing w with zeros 
w_beta = np.zeros(nvars+1) # initial value for beta
alpha = 0.0001   

## process starts
time_start = time()
for k in range(iterations):
    gradient = ridge_reg_der(w_beta,X,Y)
    for j in range(len(w_beta)): #nvars+1 = len(w_beta)
        
        unit_vector_jk = gradient[j] / np.linalg.norm(gradient) ##unit vector e_jk
        descent_direction = gradient/gradient[j]
        w_beta = w_beta - alpha*descent_direction*unit_vector_jk
        
        OF_iteration.append(ridge_reg(w_beta, X, Y)) ## calculating the OF (cost) of ridge regression with ridge_reg function
    
## process ends
time_end = time() 

## results
print('\nValues of the ridge regression coefficients - coordinated descent method:')
print('Elapsed time = %8.5f' %(time_end- time_start))
print('Objective function   = %11.5f' %OF_iteration[-1]) # the last, minimized OF.

beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-w_beta,ord=2)/np.linalg.norm(w_beta,ord=2)
print('\nBeta coefficient error = %10.5f' %beta_error)

## estimated betas
print('beta %-9s %7.3f' %('intercept',w_beta[0]))
col_list = ['X' + str(x) for x in range(0,nvars+1)]
for i in np.arange(1,nvars+1):
    print('beta %-9s %7.3f' %(col_list[i],w_beta[i]))
```

    
    Values of the ridge regression coefficients - coordinated descent method:
    Elapsed time = 60.83374
    Objective function   =  1972.13662
    
    Beta coefficient error =    0.02960
    beta intercept  -0.095
    beta X1          0.984
    beta X2         -3.976
    beta X3          0.982
    beta X4         -2.992
    beta X5          2.986
    beta X6          1.978
    beta X7          2.010
    beta X8          2.991
    beta X9         -5.005
    beta X10         0.994
    beta X11        -0.992
    beta X12        -1.010
    beta X13         0.003
    beta X14         1.971
    beta X15         1.992
    beta X16         3.989
    beta X17        -1.990
    beta X18         3.979
    beta X19        -4.002
    beta X20        -3.005
    beta X21         1.998
    beta X22         0.964
    beta X23         2.013
    beta X24        -0.019
    beta X25        -2.016
    beta X26        -1.009
    beta X27        -2.975
    beta X28        -0.975
    beta X29         2.993
    beta X30        -4.005
    beta X31         2.011
    beta X32        -3.021
    beta X33        -1.972
    beta X34        -1.013
    beta X35        -3.009
    beta X36         0.018
    beta X37        -4.988
    beta X38         4.009
    beta X39        -4.993
    beta X40        -4.012
    beta X41        -0.988
    beta X42         2.006
    beta X43        -5.008
    beta X44        -3.989
    beta X45         3.988
    beta X46         2.004
    beta X47         0.982
    beta X48        -1.995
    beta X49         0.017
    beta X50        -1.014
    beta X51        -4.020
    beta X52         0.012
    beta X53         3.013
    beta X54        -4.004
    beta X55        -3.002
    beta X56        -3.992
    beta X57        -2.014
    beta X58        -4.002
    beta X59        -3.983
    beta X60        -0.990
    beta X61        -0.009
    beta X62        -4.004
    beta X63        -2.015
    beta X64        -2.991
    beta X65        -1.986
    beta X66         3.018
    beta X67        -4.989
    beta X68         3.966
    beta X69        -2.011
    beta X70        -0.975
    beta X71        -2.021
    beta X72         2.999
    beta X73        -2.999
    beta X74         1.995
    beta X75        -4.988
    beta X76        -0.006
    beta X77        -0.002
    beta X78        -1.992
    beta X79         2.009
    beta X80         0.978
    beta X81        -0.007
    beta X82         2.026
    beta X83        -0.008
    beta X84        -2.016
    beta X85         3.010
    beta X86        -4.987
    beta X87        -3.004
    beta X88         1.004
    beta X89        -0.997
    beta X90         2.004
    beta X91        -0.966
    beta X92         3.994
    beta X93        -0.999
    beta X94        -3.995
    beta X95         2.991
    beta X96         1.015
    beta X97        -0.028
    beta X98         1.977
    beta X99        -1.995
    beta X100        2.982
    beta X101        0.990
    beta X102        3.011
    beta X103        3.980
    beta X104       -5.004
    beta X105        1.992
    beta X106        2.010
    beta X107       -0.981
    beta X108        4.000
    beta X109       -1.019
    beta X110        3.977
    beta X111        0.982
    beta X112       -0.032
    beta X113       -2.020
    beta X114       -5.014
    beta X115       -3.979
    beta X116       -0.005
    beta X117        1.992
    beta X118        2.021
    beta X119        1.987
    beta X120       -4.013
    

- As seen in the above results, the error rate has been calculated quite low which is 2%.


```python
# Plot results showing the evolution of the algorithm
plt.figure()
plt.plot(OF_iteration),plt.ylabel('Objective function'),plt.xlabel('Iterations')
plt.ticklabel_format(style='plain')    # to prevent scientific notation.
plt.show()

```


    
![png](output_41_0.png)
    


### **2.Mini-batch gradient method** 
In this step the effects of the mini-batch size in algorithm performance (number of iterations and computational time needed to reach a pre-specified tolerance limit) has been studied.

|Mini-batch gradient descent formula|
| --- |
|![image.png](attachment:4acfb020-566f-43a0-8826-c3f6620bd237.png)|

First, "mini_batch" function has been created in order to create batch samples in X and Y.


```python
# function to create a list containing mini-batches
## normally mini-batch sizes range between 50 and 1000.
def mini_batch(batch_size, X, Y): 
    mini_batches = []
    data = np.concatenate([X, Y],axis=1)
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size ##20
    
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size : (i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    
    return mini_batches
```

In this part, 5 different batch size has been tried. (30,50,300,500,1000). 


```python
print('\nRidge regression coefficients - mini-batch descent method:')

batch_sizes = [30,50,300,500,1000]
for batch_size in batch_sizes:
    ## empty lists to append inside each iteration value
    OF_iteration = list()
    tolerance_iteration = list()
    ## Initial values for the variables
    iterations = 4000 # maximum number of iterations
    alpha = 0.0001  ## initial 
    beta = np.zeros(nvars+1)
    previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y))
    precision = 0.000001 #This tells us when to stop the algorithm
    
    iter_count = 0
    ## process starts
    time_start = time()
    while (iter_count < iterations) and (previous_step_size > precision):
        mini_batches = mini_batch(batch_size, X, Y)
        for batch in mini_batches:
            X_mini, Y_mini = batch
            beta = beta - (alpha/ batch_size) * ridge_reg_der(beta, X_mini, Y_mini) 
            
            OF_iteration.append(ridge_reg(beta, X_mini, Y_mini))
            previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2)
            tolerance_iteration.append(previous_step_size)
        iter_count = iter_count + 1 ## increasing iteration number
    
    ## process ends
    time_end = time() 

    ## results
    print('\nBatch_size: ', batch_size)
    print('Elapsed time = %8.5f' %(time_end- time_start))
    print('Objective function   = %11.5f' %OF_iteration[-1]) # the last, minimized OF.
    print('Number of iterations = %5.0f' %iter_count)
    print('Optimality tolerance = %11.5f' %previous_step_size)
    beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-beta,ord=2)/np.linalg.norm(beta,ord=2)
    print('Beta coefficient error = %10.5f' %beta_error)
```

    
    Ridge regression coefficients - mini-batch descent method:
    
    Batch_size:  30
    Elapsed time = 34.39688
    Objective function   =   949.64590
    Number of iterations =  4000
    Optimality tolerance = 25770.26508
    Beta coefficient error =    0.02828
    
    Batch_size:  50
    Elapsed time = 27.62812
    Objective function   =   959.58907
    Number of iterations =  4000
    Optimality tolerance = 20420.47121
    Beta coefficient error =    0.02841
    
    Batch_size:  300
    Elapsed time = 16.48434
    Objective function   =  1301.03573
    Number of iterations =  4000
    Optimality tolerance = 11953.76960
    Beta coefficient error =    0.02927
    
    Batch_size:  500
    Elapsed time = 13.11485
    Objective function   =  1377.79658
    Number of iterations =  4000
    Optimality tolerance =  1441.89442
    Beta coefficient error =    0.02947
    
    Batch_size:  1000
    Elapsed time = 13.26293
    Objective function   =  2487.99088
    Number of iterations =  4000
    Optimality tolerance =  3384.00711
    Beta coefficient error =    0.03238
    

- After executing the mini-batch algorithm with different batch sizes, it can be definelity interpretted that when batch size increases, the execution time decreases as well. Even though none of the batch size could not converge the algortihm, which means that all of them has 4000 iterations, the optimality tolerances are different. The minimum optimality tolerance has been observed when batch size was 500. Overall, the algortihm needs more iterations according to tolerance values. Generally, OF tended to inrease when the batch size has been increased. The minimum OF has been observed 949 when the batch size was 30.

### **3. Mini-batch gradient with momentum**
In this step, the momentum(v) has been implemented to the previously obtained mini-batch algortihm. Here, apart from alpha parameter (learning rate), there is also beta parameter which is similar to acceleration in mechanics.


```python
beta_k = 0.95
v = np.zeros(nvars+1)

## empty lists to append inside each iteration value
OF_iteration = list()
tolerance_iteration = list()
## Initial values for the variables
iterations = 10000 # maximum number of iterations
alpha = 0.000001  ## initial 
beta = np.zeros(nvars+1)
previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y))
precision = 0.000001 #This tells us when to stop the algorithm
batch_size = 50

iter_count = 0
## process starts
time_start = time()
while (iter_count < iterations) and (previous_step_size > precision):
    mini_batches = mini_batch(batch_size, X, Y)
    for batch in mini_batches:
        X_mini, Y_mini = batch
        # momentum
        v = beta_k * v + (1-beta_k) * ridge_reg_der(beta, X_mini, Y_mini)
        beta = beta - alpha* v
            
        OF_iteration.append(ridge_reg(beta, X_mini, Y_mini))
        previous_step_size = np.linalg.norm(ridge_reg_der(beta,X,Y),ord=2)
        tolerance_iteration.append(previous_step_size)
    iter_count = iter_count + 1 ## increasing iteration number
    
## process ends
time_end = time() 

## results
print('\nBatch_size: ', batch_size)
print('Elapsed time = %8.5f' %(time_end- time_start))
print('Objective function   = %11.5f' %OF_iteration[-1]) # the last, minimized OF.
print('Number of iterations = %5.0f' %iter_count)
print('Optimality tolerance = %11.5f' %previous_step_size)
beta_error = np.linalg.norm(np.transpose(beta_ridge_exact)-beta,ord=2)/np.linalg.norm(beta,ord=2)
print('Beta coefficient error = %10.5f' %beta_error)
```

    
    Batch_size:  50
    Elapsed time = 107.31314
    Objective function   =   957.94615
    Number of iterations = 10000
    Optimality tolerance =  1615.12554
    Beta coefficient error =    0.02812
    

## f) Consider the constrained problem:

$$ \min_{\beta} ||y-X\beta||^2_2$$

$$ st: \sum_{i=1}^{K} \beta_{i} = 1 $$

Estimate the optimal value of the regression coefficients in (1) by implementing a penalization algorithm


Here the penalization which is $\sum_{i=1}^{K} \beta_{i} = 1 $ added to regression function. After creating the penalization function, the gradient of the function which is the derivative of the penalization function has been created in order to minimize with BFGS method.


```python
# penalization function
def penalization(beta_pn,X,Y):
    rho = 1
    beta_pn = np.matrix(beta_pn)
    z = Y - np.dot(X,beta_pn.T)
    val_f = np.dot(z.T,z) + 1*np.dot(beta_pn - 1,(beta_pn - 1).T)
    val_ff = np.squeeze(np.asarray(val_f))
    return val_ff # OF

# gradient function
def penalization_der(beta_pn,X,Y):
    beta_pn = np.matrix(beta_pn)
    gradient = -2*np.dot((Y-np.dot(X,(beta_pn).T)).T,X) + 2 * 1 * beta_pn -2
    aa = np.squeeze(np.asarray(gradient))
    return aa
from scipy.optimize import minimize

beta_ls = np.zeros(nvars+1)

time_start = time()
res = minimize(penalization, beta_ls, args=(X, Y),  jac=penalization_der, hess=None, method='BFGS', options={'disp': True})
time_elapsed = (time() - time_start)

# Print results
print('\nValues of the least squares coefficients obtained with Nelder-Mead:')
for i in range(nvars+1):
    print('beta %3d %7.3f' %(i,res.x[i]))
```

    Optimization terminated successfully.
             Current function value: 2072.526910
             Iterations: 141
             Function evaluations: 257
             Gradient evaluations: 257
    
    Values of the least squares coefficients obtained with Nelder-Mead:
    beta   0  -0.763
    beta   1   1.000
    beta   2  -3.980
    beta   3   0.986
    beta   4  -2.995
    beta   5   3.000
    beta   6   1.982
    beta   7   2.011
    beta   8   2.995
    beta   9  -5.005
    beta  10   0.993
    beta  11  -0.999
    beta  12  -1.011
    beta  13   0.008
    beta  14   1.973
    beta  15   1.990
    beta  16   3.997
    beta  17  -1.983
    beta  18   3.982
    beta  19  -3.999
    beta  20  -3.008
    beta  21   1.997
    beta  22   0.980
    beta  23   2.016
    beta  24  -0.018
    beta  25  -2.013
    beta  26  -0.997
    beta  27  -2.978
    beta  28  -0.974
    beta  29   2.999
    beta  30  -4.007
    beta  31   2.017
    beta  32  -3.028
    beta  33  -1.971
    beta  34  -1.012
    beta  35  -3.009
    beta  36   0.013
    beta  37  -4.980
    beta  38   4.009
    beta  39  -4.997
    beta  40  -4.016
    beta  41  -0.992
    beta  42   2.004
    beta  43  -5.003
    beta  44  -3.996
    beta  45   3.996
    beta  46   2.006
    beta  47   0.992
    beta  48  -2.004
    beta  49   0.021
    beta  50  -1.009
    beta  51  -4.021
    beta  52   0.008
    beta  53   3.008
    beta  54  -4.003
    beta  55  -3.008
    beta  56  -3.996
    beta  57  -2.008
    beta  58  -4.007
    beta  59  -3.988
    beta  60  -0.989
    beta  61  -0.011
    beta  62  -4.003
    beta  63  -2.005
    beta  64  -2.995
    beta  65  -1.980
    beta  66   3.015
    beta  67  -4.988
    beta  68   3.979
    beta  69  -2.008
    beta  70  -0.975
    beta  71  -2.010
    beta  72   3.009
    beta  73  -3.001
    beta  74   1.997
    beta  75  -4.991
    beta  76  -0.003
    beta  77  -0.003
    beta  78  -1.996
    beta  79   2.013
    beta  80   0.988
    beta  81  -0.006
    beta  82   2.032
    beta  83  -0.016
    beta  84  -2.018
    beta  85   3.015
    beta  86  -4.991
    beta  87  -3.007
    beta  88   1.005
    beta  89  -0.994
    beta  90   2.003
    beta  91  -0.964
    beta  92   4.001
    beta  93  -1.001
    beta  94  -4.001
    beta  95   3.005
    beta  96   1.010
    beta  97  -0.018
    beta  98   1.986
    beta  99  -1.996
    beta 100   2.988
    beta 101   0.999
    beta 102   3.007
    beta 103   3.991
    beta 104  -5.010
    beta 105   1.999
    beta 106   2.012
    beta 107  -0.990
    beta 108   4.017
    beta 109  -1.013
    beta 110   3.981
    beta 111   0.985
    beta 112  -0.032
    beta 113  -2.013
    beta 114  -5.008
    beta 115  -3.991
    beta 116   0.001
    beta 117   1.995
    beta 118   2.009
    beta 119   1.988
    beta 120  -4.009
    
