# Regularization

**TODO:** Introduce regularization as a concept.

We will use the `Hitters` dataset from the `ISLR` package to explore two shrinkage methods: **ridge** and **lasso**. These are otherwise known as **penalized regression** methods.


```r
data(Hitters, package = "ISLR")
```

This dataset has some missing data in the response `Salaray`. We use the `na.omit()` function the clean the dataset.


```r
sum(is.na(Hitters))
```

```
## [1] 59
```

```r
sum(is.na(Hitters$Salary))
```

```
## [1] 59
```

```r
Hitters = na.omit(Hitters)
sum(is.na(Hitters))
```

```
## [1] 0
```

The predictors variables are offensive and defensive statistics for a number of baseball players.


```r
names(Hitters)
```

```
##  [1] "AtBat"     "Hits"      "HmRun"     "Runs"      "RBI"      
##  [6] "Walks"     "Years"     "CAtBat"    "CHits"     "CHmRun"   
## [11] "CRuns"     "CRBI"      "CWalks"    "League"    "Division" 
## [16] "PutOuts"   "Assists"   "Errors"    "Salary"    "NewLeague"
```

We use the `glmnet()` and `cv.glmnet()` functions in the `glmnet` package to fit penalized regressions.


```r
# this is a temporary workaround for an issue with glmnet, Matrix, and R version 3.3.3
# see here: http://stackoverflow.com/questions/43282720/r-error-in-validobject-object-when-running-as-script-but-not-in-console
library(methods)
```


```r
library(glmnet)
```

The `glmnet` function does not allow the use of model formulas, so we setup the data for ease of use with `glmnet`.


```r
X = model.matrix(Salary ~ ., Hitters)[, -1]
y = Hitters$Salary
```

First, we fit a regular linear regression, and note the size of the predictors' coefficients, and predictors' coefficients squared. (The two penalties we will use.)


```r
fit = lm(Salary ~ ., Hitters)
coef(fit)
```

```
##  (Intercept)        AtBat         Hits        HmRun         Runs 
##  163.1035878   -1.9798729    7.5007675    4.3308829   -2.3762100 
##          RBI        Walks        Years       CAtBat        CHits 
##   -1.0449620    6.2312863   -3.4890543   -0.1713405    0.1339910 
##       CHmRun        CRuns         CRBI       CWalks      LeagueN 
##   -0.1728611    1.4543049    0.8077088   -0.8115709   62.5994230 
##    DivisionW      PutOuts      Assists       Errors   NewLeagueN 
## -116.8492456    0.2818925    0.3710692   -3.3607605  -24.7623251
```

```r
sum(abs(coef(fit)[-1]))
```

```
## [1] 238.7295
```

```r
sum(coef(fit)[-1] ^ 2)
```

```
## [1] 18337.3
```


## Ridge Regression

We first illustrate **ridge regression**, which can be fit using `glmnet()` with `alpha = 0` and seeks to minimize

$$
\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}    \right) ^ 2 + \lambda \sum_{j=1}^{p} \beta_j^2 .
$$

Notice that the intercept is **not** penalized. Also, note that that ridge regression is **not** scale invariant like the usual unpenalized regression. Thankfully, `glmnet()` takes care of this internally. It automatically standardizes input for fitting, then reports fitted coefficient using the original scale.

The two plots illustrate how much the coefficients are penalized for different values of $\lambda$. Notice none of the coefficients are forced to be zero.


```r
fit_ridge = glmnet(X, y, alpha = 0)
plot(fit_ridge)
```

![](24-regularization_files/figure-latex/ridge-1.pdf)<!-- --> 

```r
plot(fit_ridge, xvar = "lambda", label = TRUE)
```

![](24-regularization_files/figure-latex/ridge-2.pdf)<!-- --> 

```r
dim(coef(fit_ridge))
```

```
## [1]  20 100
```

We use cross-validation to select a good $\lambda$ value. The `cv.glmnet()`function uses 10 folds by default. The plot illustrates the MSE for the $\lambda$s considered. Two lines are drawn. The first is the $\lambda$ that gives the smallest MSE. The second is the $\lambda$ that gives an MSE within one standard error of the smallest.


```r
fit_ridge_cv = cv.glmnet(X, y, alpha = 0)
plot(fit_ridge_cv)
```

![](24-regularization_files/figure-latex/unnamed-chunk-8-1.pdf)<!-- --> 

The `cv.glmnet()` function returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are smaller than the full linear regression. (As we would expect.)


```r
coef(fit_ridge_cv)
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 268.287904048
## AtBat         0.075738253
## Hits          0.300154606
## HmRun         1.022784256
## Runs          0.489474365
## RBI           0.495632199
## Walks         0.626356706
## Years         2.143185629
## CAtBat        0.006369369
## CHits         0.024201921
## CHmRun        0.180499284
## CRuns         0.048544437
## CRBI          0.050169414
## CWalks        0.049897906
## LeagueN       1.802540422
## DivisionW   -16.185025138
## PutOuts       0.040146198
## Assists       0.005930000
## Errors       -0.087618226
## NewLeagueN    1.836629079
```

```r
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  1.226775e+01
## AtBat       -2.419845e-02
## Hits         1.160708e+00
## HmRun       -2.631285e-01
## Runs         1.143697e+00
## RBI          8.628801e-01
## Walks        1.966769e+00
## Years       -1.136139e+00
## CAtBat       1.065732e-02
## CHits        7.226563e-02
## CHmRun       4.922058e-01
## CRuns        1.434083e-01
## CRBI         1.528435e-01
## CWalks       9.709707e-04
## LeagueN      3.150933e+01
## DivisionW   -1.003989e+02
## PutOuts      2.096675e-01
## Assists      5.665892e-02
## Errors      -2.194190e+00
## NewLeagueN   4.428640e+00
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 11106.17
```

```r
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 268.287904048
## AtBat         0.075738253
## Hits          0.300154606
## HmRun         1.022784256
## Runs          0.489474365
## RBI           0.495632199
## Walks         0.626356706
## Years         2.143185629
## CAtBat        0.006369369
## CHits         0.024201921
## CHmRun        0.180499284
## CRuns         0.048544437
## CRBI          0.050169414
## CWalks        0.049897906
## LeagueN       1.802540422
## DivisionW   -16.185025138
## PutOuts       0.040146198
## Assists       0.005930000
## Errors       -0.087618226
## NewLeagueN    1.836629079
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 275.24
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 141009.7
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.9168 450.0010 449.4470 449.1758 448.8793 448.5554 448.2016
##  [8] 447.8153 447.3938 446.9341 446.4331 445.8874 445.2934 444.6474
## [15] 443.9454 443.1832 442.3566 441.4612 440.4923 439.4453 438.3155
## [22] 437.0982 435.7888 434.3830 432.8765 431.2655 429.5468 427.7175
## [29] 425.7759 423.7206 421.5518 419.2706 416.8794 414.3821 411.7843
## [36] 409.0930 406.3173 403.4671 400.5547 397.5937 394.5990 391.5867
## [43] 388.5736 385.5770 382.6144 379.7027 376.8585 374.0972 371.4329
## [50] 368.8775 366.4403 364.1306 361.9544 359.9153 358.0151 356.2536
## [57] 354.6288 353.1373 351.7744 350.5345 349.4111 348.3976 347.4864
## [64] 346.6703 345.9435 345.3007 344.7367 344.2400 343.8050 343.4312
## [71] 343.1136 342.8420 342.6149 342.4291 342.2785 342.1627 342.0745
## [78] 342.0108 341.9665 341.9458 341.9397 341.9450 341.9613 341.9886
## [85] 342.0186 342.0545 342.0953 342.1304 342.1703 342.2017 342.2456
## [92] 342.2611 342.3091 342.3117 342.3537 342.3475 342.3827 342.3711
## [99] 342.4025
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.9397
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 379.7027
```


## Lasso

We now illustrate **lasso**, which can be fit using `glmnet()` with `alpha = 1` and seeks to minimize

$$
\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}    \right) ^ 2 + \lambda \sum_{j=1}^{p} |\beta_j| .
$$

Like ridge, lasso is not scale invariant.

The two plots illustrate how much the coefficients are penalized for different values of $\lambda$. Notice some of the coefficients are forced to be zero.


```r
fit_lasso = glmnet(X, y, alpha = 1)
plot(fit_lasso)
```

![](24-regularization_files/figure-latex/lasso-1.pdf)<!-- --> 

```r
plot(fit_lasso, xvar = "lambda", label = TRUE)
```

![](24-regularization_files/figure-latex/lasso-2.pdf)<!-- --> 

```r
dim(coef(fit_lasso))
```

```
## [1] 20 80
```

Again, to actually pick a $\lambda$, we will use cross-validation. The plot is similar to the ridge plot. Notice along the top is the number of features in the model. (Which changed in this plot.)


```r
fit_lasso_cv = cv.glmnet(X, y, alpha = 1)
plot(fit_lasso_cv)
```

![](24-regularization_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

`cv.glmnet()` returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are again smaller than the full linear regression. (As we would expect.) Some coefficients are 0.


```r
coef(fit_lasso_cv)
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 193.74263858
## AtBat         .         
## Hits          1.21471320
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.28957902
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.12923755
## CRBI          0.31515925
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.02533305
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
coef(fit_lasso_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept)  129.4155571
## AtBat         -1.6130155
## Hits           5.8058915
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.8469340
## Years         -9.9724045
## CAtBat         .        
## CHits          .        
## CHmRun         0.5374550
## CRuns          0.6811938
## CRBI           0.3903563
## CWalks        -0.5560144
## LeagueN       32.4646094
## DivisionW   -119.3480842
## PutOuts        0.2741895
## Assists        0.1855978
## Errors        -2.1650837
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 178.8408
```

```r
coef(fit_lasso_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 193.74263858
## AtBat         .         
## Hits          1.21471320
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.28957902
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.12923755
## CRBI          0.31515925
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.02533305
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 2.974022
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 127112.4
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 450.2818 441.4651 431.1229 422.5383 414.4704 406.2837 398.8817
##  [8] 391.7275 385.2637 380.0246 375.5900 371.9386 368.3817 365.0250
## [15] 362.1567 359.1181 356.3796 354.0468 351.9612 350.1176 348.5747
## [22] 347.2972 346.2283 345.3366 344.5953 343.9914 343.5870 343.4716
## [29] 343.5235 343.6428 343.7996 344.0597 344.3855 344.7183 345.0273
## [36] 345.3890 345.8085 346.2003 346.5051 346.1877 345.4803 344.8495
## [43] 344.4175 344.0582 343.7548 343.4779 343.3719 343.4321 343.3785
## [50] 343.2646 343.2576 343.2894 343.3346 343.4079 343.4886 343.6225
## [57] 343.7985 344.0277 344.1966 344.2745 344.4175 344.5793 344.7782
## [64] 344.9560 345.1265 345.3089 345.4647 345.6447 345.7638 345.9428
## [71] 346.0532
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 343.2576
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.9386
```

## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
#fit_lasso_cv
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high  conf.low nzero
## 1  255.2820965 202753.7  30936.50  233690.2 171817.15     0
## 2  232.6035386 194891.4  30740.37  225631.8 164151.06     1
## 3  211.9396813 185867.0  29603.41  215470.4 156263.58     2
## 4  193.1115442 178538.6  28714.06  207252.7 149824.57     2
## 5  175.9560468 171785.7  28111.89  199897.6 143673.79     3
## 6  160.3245966 165066.5  27778.52  192845.0 137287.97     4
## 7  146.0818013 159106.6  27522.46  186629.0 131584.12     4
## 8  133.1042967 153450.5  26831.65  180282.1 126618.82     4
## 9  121.2796778 148428.1  26008.48  174436.6 122419.67     4
## 10 110.5055255 144418.7  25352.51  169771.2 119066.22     4
## 11 100.6885192 141067.9  24805.35  165873.2 116262.53     5
## 12  91.7436287 138338.3  24391.53  162729.8 113946.78     5
## 13  83.5933775 135705.1  24117.86  159823.0 111587.25     5
## 14  76.1671723 133243.2  23916.28  157159.5 109326.96     5
## 15  69.4006906 131157.5  23745.75  154903.3 107411.74     6
## 16  63.2353245 128965.8  23551.27  152517.0 105414.50     6
## 17  57.6176726 127006.4  23375.71  150382.2 103630.73     6
## 18  52.4990774 125349.1  23220.63  148569.8 102128.50     6
## 19  47.8352040 123876.7  23031.81  146908.5 100844.89     6
## 20  43.5856563 122582.3  22827.33  145409.6  99754.97     6
## 21  39.7136268 121504.3  22651.08  144155.4  98853.21     6
## 22  36.1855776 120615.3  22499.06  143114.4  98116.28     6
## 23  32.9709506 119874.0  22367.52  142241.5  97506.51     6
## 24  30.0419022 119257.4  22254.13  141511.5  97003.24     6
## 25  27.3730624 118745.9  22157.24  140903.2  96588.69     6
## 26  24.9413150 118330.1  22072.32  140402.4  96257.75     6
## 27  22.7255973 118052.0  22003.29  140055.3  96048.71     6
## 28  20.7067179 117972.8  21997.74  139970.5  95975.02     6
## 29  18.8671902 118008.4  22046.99  140055.4  95961.42     6
## 30  17.1910810 118090.4  22095.62  140186.0  95994.76     7
## 31  15.6638727 118198.2  22144.96  140343.1  96053.22     7
## 32  14.2723374 118377.1  22224.38  140601.5  96152.73     7
## 33  13.0044223 118601.4  22325.42  140926.8  96275.96     9
## 34  11.8491453 118830.7  22424.12  141254.8  96406.56     9
## 35  10.7964999 119043.9  22516.05  141559.9  96527.81     9
## 36   9.8373686 119293.5  22593.13  141886.7  96700.41     9
## 37   8.9634439 119583.5  22648.74  142232.2  96934.75     9
## 38   8.1671562 119854.7  22683.24  142537.9  97171.42    11
## 39   7.4416086 120065.8  22669.74  142735.6  97396.08    11
## 40   6.7805166 119846.0  22582.44  142428.4  97263.52    12
## 41   6.1781542 119356.7  22367.74  141724.4  96988.92    12
## 42   5.6293040 118921.1  22123.29  141044.4  96797.86    13
## 43   5.1292121 118623.4  21913.30  140536.7  96710.09    13
## 44   4.6735471 118376.1  21712.50  140088.6  96663.57    13
## 45   4.2583620 118167.3  21500.23  139667.6  96667.11    13
## 46   3.8800609 117977.0  21299.96  139277.0  96677.08    13
## 47   3.5353670 117904.2  21133.35  139037.6  96770.90    13
## 48   3.2212947 117945.6  21030.91  138976.5  96914.73    13
## 49   2.9351238 117908.8  20892.83  138801.6  97015.96    13
## 50   2.6743755 117830.6  20740.65  138571.2  97089.93    13
## 51   2.4367913 117825.8  20595.33  138421.1  97230.43    13
## 52   2.2203135 117847.6  20449.20  138296.8  97398.44    14
## 53   2.0230670 117878.6  20317.71  138196.4  97560.94    15
## 54   1.8433433 117929.0  20198.08  138127.1  97730.89    15
## 55   1.6795857 117984.4  20082.82  138067.3  97901.62    17
## 56   1.5303760 118076.4  19977.09  138053.5  98099.36    17
## 57   1.3944216 118197.4  19891.42  138088.8  98306.00    17
## 58   1.2705450 118355.1  19829.84  138184.9  98525.24    17
## 59   1.1576733 118471.3  19792.68  138264.0  98678.60    17
## 60   1.0548288 118524.9  19749.30  138274.2  98775.64    17
## 61   0.9611207 118623.4  19705.69  138329.1  98917.71    17
## 62   0.8757374 118734.9  19675.45  138410.4  99059.46    17
## 63   0.7979393 118872.0  19651.63  138523.6  99220.37    17
## 64   0.7270526 118994.6  19634.23  138628.8  99360.38    17
## 65   0.6624632 119112.3  19619.76  138732.0  99492.52    18
## 66   0.6036118 119238.2  19604.85  138843.1  99633.39    18
## 67   0.5499886 119345.8  19594.99  138940.8  99750.86    18
## 68   0.5011291 119470.3  19593.98  139064.3  99876.31    17
## 69   0.4566102 119552.6  19581.33  139133.9  99971.28    18
## 70   0.4160462 119676.4  19578.93  139255.3 100097.47    18
## 71   0.3790858 119752.8  19564.11  139316.9 100188.72    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   91.74363
```


## Simulation Study, p > n

Aside from simply shrinking coefficients (ridge) and setting some coefficients to 0 (lasso), penalized regression also has the advantage of being able to handle the $p > n$ case.


```r
set.seed(1234)
n = 1000
p = 5500
X = replicate(p, rnorm(n = n))
beta = c(1, 1, 1, rep(0, 5497))
z = X %*% beta
prob = exp(z) / (1 + exp(z))
y = as.factor(rbinom(length(z), size = 1, prob = prob))
```

We first simulate a classification example where $p > n$.


```r
# glm(y ~ X, family = "binomial")
# will not converge
```

We then use a lasso penalty to fit penalized logistic regression. This minimizes

$$
\sum_{i=1}^{n} L\left(y_i, \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}\right) + \lambda \sum_{j=1}^{p} |\beta_j|
$$

where $L$ is the appropriate *negative* **log**-likelihood.


```r
library(glmnet)
fit_cv = cv.glmnet(X, y, family = "binomial", alpha = 1)
plot(fit_cv)
```

![](24-regularization_files/figure-latex/unnamed-chunk-15-1.pdf)<!-- --> 


```r
head(coef(fit_cv), n = 10)
```

```
## 10 x 1 sparse Matrix of class "dgCMatrix"
##                      1
## (Intercept) 0.02397452
## V1          0.59674958
## V2          0.56251761
## V3          0.60065105
## V4          .         
## V5          .         
## V6          .         
## V7          .         
## V8          .         
## V9          .
```


```r
fit_cv$nzero
```

```
##  s0  s1  s2  s3  s4  s5  s6  s7  s8  s9 s10 s11 s12 s13 s14 s15 s16 s17 
##   0   2   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 
## s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 s34 s35 
##   3   3   3   3   3   3   3   3   3   3   3   3   4   6   7  10  18  24 
## s36 s37 s38 s39 s40 s41 s42 s43 s44 s45 s46 s47 s48 s49 s50 s51 s52 s53 
##  35  54  65  75  86 100 110 129 147 168 187 202 221 241 254 269 283 298 
## s54 s55 s56 s57 s58 s59 s60 s61 s62 s63 s64 s65 s66 s67 s68 s69 s70 s71 
## 310 324 333 350 364 375 387 400 411 429 435 445 453 455 462 466 475 481 
## s72 s73 s74 s75 s76 s77 s78 s79 s80 s81 s82 s83 s84 s85 s86 s87 s88 s89 
## 487 491 496 498 502 504 512 518 523 526 528 536 543 550 559 561 563 566 
## s90 s91 s92 s93 s94 s95 s96 s97 s98 
## 570 571 576 582 586 590 596 596 600
```

Notice, only the first three predictors generated are truly significant, and that is exactly what the suggested model finds.


```r
fit_1se = glmnet(X, y, family = "binomial", lambda = fit_cv$lambda.1se)
which(as.vector(as.matrix(fit_1se$beta)) != 0)
```

```
## [1] 1 2 3
```

We can also see in the following plots, the three features entering the model well ahead of the irrelevant features.


```r
plot(glmnet(X, y, family = "binomial"))
```

![](24-regularization_files/figure-latex/unnamed-chunk-19-1.pdf)<!-- --> 


```r
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```

![](24-regularization_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 

We can extract the two relevant $\lambda$ values.


```r
fit_cv$lambda.min
```

```
## [1] 0.03087158
```

```r
fit_cv$lambda.1se
```

```
## [1] 0.0514969
```

Since `cv.glmnet()` does not calculate prediction accuracy for classification, we take the $\lambda$ values and create a grid for `caret` to search in order to obtain prediction accuracy with `train()`. We set $\alpha = 1$ in this grid, as `glmnet` can actually tune over the $\alpha = 1$ parameter. (More on that later.)

Note that we have to force `y` to be a factor, so that `train()` recognizes we want to have a binomial response. The `train()` function in `caret` use the type of variable in `y` to determine if you want to use `family = "binomial"` or `family = "gaussian"`.


```r
library(caret)
cv_5 = trainControl(method = "cv", number = 5)
lasso_grid = expand.grid(alpha = 1, 
                         lambda = c(fit_cv$lambda.min, fit_cv$lambda.1se))
lasso_grid
```

```
##   alpha     lambda
## 1     1 0.03087158
## 2     1 0.05149690
```


```r
sim_data = data.frame(y, X)
fit_lasso = train(
  y ~ ., data = sim_data,
  method = "glmnet",
  trControl = cv_5,
  tuneGrid = lasso_grid
)
fit_lasso$results
```

```
##   alpha     lambda  Accuracy     Kappa AccuracySD    KappaSD
## 1     1 0.03087158 0.7679304 0.5358028 0.03430230 0.06844656
## 2     1 0.05149690 0.7689003 0.5377583 0.02806941 0.05596114
```


## External Links

- [`glmnet` Web Vingette](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html) - Details from the package developers.


## RMarkdown

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.4.1 and the following packages:

- Base Packages, Attached


```
## [1] "methods"   "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "base"
```

- Additional Packages, Attached


```
## [1] "caret"   "ggplot2" "lattice" "broom"   "glmnet"  "foreach" "Matrix"
```

- Additional Packages, Not Attached


```
##  [1] "Rcpp"         "lubridate"    "tidyr"        "class"       
##  [5] "assertthat"   "rprojroot"    "digest"       "ipred"       
##  [9] "psych"        "R6"           "plyr"         "backports"   
## [13] "stats4"       "evaluate"     "e1071"        "rlang"       
## [17] "lazyeval"     "kernlab"      "rpart"        "rmarkdown"   
## [21] "splines"      "CVST"         "ddalpha"      "gower"       
## [25] "stringr"      "foreign"      "munsell"      "compiler"    
## [29] "pkgconfig"    "mnormt"       "dimRed"       "htmltools"   
## [33] "nnet"         "tibble"       "prodlim"      "DRR"         
## [37] "bookdown"     "codetools"    "RcppRoll"     "dplyr"       
## [41] "withr"        "MASS"         "recipes"      "ModelMetrics"
## [45] "grid"         "nlme"         "gtable"       "magrittr"    
## [49] "scales"       "stringi"      "reshape2"     "bindrcpp"    
## [53] "timeDate"     "robustbase"   "lava"         "iterators"   
## [57] "tools"        "glue"         "DEoptimR"     "purrr"       
## [61] "parallel"     "survival"     "yaml"         "colorspace"  
## [65] "knitr"        "bindr"
```
