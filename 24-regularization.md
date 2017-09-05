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
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
```

```r
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  7.645824e+01
## AtBat       -6.315180e-01
## Hits         2.642160e+00
## HmRun       -1.388233e+00
## Runs         1.045729e+00
## RBI          7.315713e-01
## Walks        3.278001e+00
## Years       -8.723734e+00
## CAtBat       1.256354e-04
## CHits        1.318975e-01
## CHmRun       6.895578e-01
## CRuns        2.830055e-01
## CRBI         2.514905e-01
## CWalks      -2.599851e-01
## LeagueN      5.233720e+01
## DivisionW   -1.224170e+02
## PutOuts      2.623667e-01
## Assists      1.629044e-01
## Errors      -3.644002e+00
## NewLeagueN  -1.702598e+01
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 18126.85
```

```r
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 588.9958
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 130404.9
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.8048 449.8194 449.2989 449.0290 448.7340 448.4117 448.0596
##  [8] 447.6752 447.2557 446.7983 446.2997 445.7566 445.1654 444.5225
## [15] 443.8239 443.0652 442.2425 441.3513 440.3868 439.3446 438.2199
## [22] 437.0080 435.7044 434.3047 432.8047 431.2007 429.4892 427.6676
## [29] 425.7340 423.6871 421.5271 419.2550 416.8733 414.3859 411.7982
## [36] 409.1176 406.3528 403.5140 400.6135 397.6649 394.6832 391.6845
## [43] 388.6858 385.7044 382.7578 379.8634 377.0377 374.2961 371.6528
## [50] 369.1195 366.7068 364.4229 362.2739 360.2635 358.3933 356.6631
## [57] 355.0706 353.6123 352.2832 351.0773 349.9880 349.0081 348.1302
## [64] 347.3461 346.6512 346.0398 345.5005 345.0277 344.6158 344.2628
## [71] 343.9572 343.6973 343.4808 343.2975 343.1441 343.0213 342.9217
## [78] 342.8436 342.7814 342.7339 342.6969 342.6687 342.6480 342.6303
## [85] 342.6166 342.6026 342.5890 342.5732 342.5552 342.5344 342.5112
## [92] 342.4838 342.4541 342.4213 342.3872 342.3511 342.3149 342.2791
## [99] 342.2458
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.2458
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 366.7068
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
##                         1
## (Intercept)  153.26245052
## AtBat         -1.93059099
## Hits           6.80153148
## HmRun          1.30268871
## Runs          -1.03744367
## RBI            .         
## Walks          5.62556821
## Years         -7.58334496
## CAtBat        -0.06800322
## CHits          .         
## CHmRun         0.21277161
## CRuns          1.13540300
## CRBI           0.57729486
## CWalks        -0.73080821
## LeagueN       47.11254249
## DivisionW   -116.63369442
## PutOuts        0.28199476
## Assists        0.29084052
## Errors        -2.87461295
## NewLeagueN   -10.65739592
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 204.8565
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
##  [1] 449.9132 440.8377 430.5613 421.9740 414.0219 405.9902 397.8838
##  [8] 390.7803 384.5943 379.2731 374.7152 370.6438 367.0556 363.9368
## [15] 360.9792 358.2465 355.6852 353.4485 351.4717 349.7810 348.3035
## [22] 347.0462 345.9816 345.0932 344.3707 343.8624 343.5768 343.3821
## [29] 343.2705 343.2113 343.2077 343.2228 343.2309 343.2383 343.3346
## [36] 343.7628 344.3938 344.6231 344.4400 344.0368 343.3955 342.6086
## [43] 341.7908 341.1184 340.3840 339.6654 339.0417 338.5011 338.0763
## [50] 337.7466 337.4821 337.2845 337.1894 337.1403 337.0446 336.9227
## [57] 336.7486 336.6216 336.4954 336.4089 336.3800 336.3888 336.4125
## [64] 336.5205 336.6132 336.7223 336.8155 336.9133 336.9680 337.0629
## [71] 337.1508
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.38
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.6438
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
## 1  255.2820965 202421.9  28171.95  230593.8 174249.92     0
## 2  232.6035386 194337.9  28465.82  222803.7 165872.08     1
## 3  211.9396813 185383.0  27733.39  213116.4 157649.60     2
## 4  193.1115442 178062.0  27150.02  205212.1 150912.02     2
## 5  175.9560468 171414.1  26801.89  198216.0 144612.22     3
## 6  160.3245966 164828.1  26462.41  191290.5 138365.65     4
## 7  146.0818013 158311.5  25766.63  184078.2 132544.89     4
## 8  133.1042967 152709.2  25219.37  177928.6 127489.87     4
## 9  121.2796778 147912.8  24825.61  172738.4 123087.17     4
## 10 110.5055255 143848.1  24515.05  168363.1 119333.01     4
## 11 100.6885192 140411.5  24321.31  164732.8 116090.15     5
## 12  91.7436287 137376.8  24203.27  161580.1 113173.57     5
## 13  83.5933775 134729.8  24150.08  158879.9 110579.70     5
## 14  76.1671723 132450.0  24126.07  156576.1 108323.92     5
## 15  69.4006906 130306.0  24120.23  154426.2 106185.74     6
## 16  63.2353245 128340.6  24096.73  152437.3 104243.84     6
## 17  57.6176726 126511.9  23995.02  150507.0 102516.92     6
## 18  52.4990774 124925.9  23895.27  148821.1 101030.60     6
## 19  47.8352040 123532.3  23797.51  147329.9  99734.82     6
## 20  43.5856563 122346.7  23716.75  146063.5  98629.99     6
## 21  39.7136268 121315.3  23642.15  144957.5  97673.19     6
## 22  36.1855776 120441.0  23579.84  144020.9  96861.20     6
## 23  32.9709506 119703.3  23528.93  143232.2  96174.37     6
## 24  30.0419022 119089.3  23492.19  142581.5  95597.10     6
## 25  27.3730624 118591.2  23471.19  142062.4  95120.02     6
## 26  24.9413150 118241.4  23472.02  141713.4  94769.34     6
## 27  22.7255973 118045.0  23483.91  141528.9  94561.11     6
## 28  20.7067179 117911.3  23495.04  141406.3  94416.26     6
## 29  18.8671902 117834.6  23508.91  141343.5  94325.71     6
## 30  17.1910810 117794.0  23523.73  141317.7  94270.27     7
## 31  15.6638727 117791.5  23541.60  141333.1  94249.90     7
## 32  14.2723374 117801.9  23564.51  141366.4  94237.40     7
## 33  13.0044223 117807.5  23594.14  141401.6  94213.32     9
## 34  11.8491453 117812.5  23620.16  141432.7  94192.39     9
## 35  10.7964999 117878.6  23654.30  141532.9  94224.32     9
## 36   9.8373686 118172.9  23816.58  141989.5  94356.31     9
## 37   8.9634439 118607.1  24061.22  142668.3  94545.89     9
## 38   8.1671562 118765.1  24258.12  143023.2  94506.93    11
## 39   7.4416086 118638.9  24395.79  143034.7  94243.11    11
## 40   6.7805166 118361.3  24535.83  142897.1  93825.47    12
## 41   6.1781542 117920.5  24663.80  142584.3  93256.65    12
## 42   5.6293040 117380.6  24755.71  142136.3  92624.93    13
## 43   5.1292121 116821.0  24851.24  141672.2  91969.74    13
## 44   4.6735471 116361.8  24958.79  141320.6  91403.00    13
## 45   4.2583620 115861.2  25002.47  140863.7  90858.77    13
## 46   3.8800609 115372.6  25021.64  140394.2  90350.94    13
## 47   3.5353670 114949.3  25041.54  139990.8  89907.73    13
## 48   3.2212947 114583.0  25062.45  139645.5  89520.56    13
## 49   2.9351238 114295.6  25084.54  139380.1  89211.06    13
## 50   2.6743755 114072.8  25105.08  139177.8  88967.67    13
## 51   2.4367913 113894.2  25127.09  139021.2  88767.06    13
## 52   2.2203135 113760.9  25150.52  138911.4  88610.34    14
## 53   2.0230670 113696.7  25174.20  138870.9  88522.47    15
## 54   1.8433433 113663.5  25204.01  138867.6  88459.54    15
## 55   1.6795857 113599.0  25230.12  138829.2  88368.92    17
## 56   1.5303760 113516.9  25268.49  138785.4  88248.40    17
## 57   1.3944216 113399.6  25281.58  138681.2  88118.02    17
## 58   1.2705450 113314.1  25305.55  138619.7  88008.58    17
## 59   1.1576733 113229.2  25311.98  138541.2  87917.19    17
## 60   1.0548288 113170.9  25320.13  138491.0  87850.79    17
## 61   0.9611207 113151.5  25339.95  138491.5  87811.57    17
## 62   0.8757374 113157.4  25365.77  138523.2  87791.64    17
## 63   0.7979393 113173.3  25395.23  138568.6  87778.12    17
## 64   0.7270526 113246.0  25435.36  138681.4  87810.68    17
## 65   0.6624632 113308.4  25469.03  138777.5  87839.39    18
## 66   0.6036118 113381.9  25501.06  138883.0  87880.84    18
## 67   0.5499886 113444.7  25528.67  138973.3  87916.00    18
## 68   0.5011291 113510.6  25553.91  139064.5  87956.66    17
## 69   0.4566102 113547.4  25578.90  139126.3  87968.52    18
## 70   0.4160462 113611.4  25597.88  139209.3  88013.50    18
## 71   0.3790858 113670.6  25621.86  139292.5  88048.79    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1  0.9611207   91.74363
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
## 1     1 0.03087158 0.7609903 0.5218887 0.01486223 0.03000986
## 2     1 0.05149690 0.7659604 0.5319189 0.01807380 0.03594319
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
##  [1] "reshape2"     "splines"      "colorspace"   "htmltools"   
##  [5] "stats4"       "yaml"         "mgcv"         "rlang"       
##  [9] "ModelMetrics" "e1071"        "nloptr"       "foreign"     
## [13] "glue"         "bindrcpp"     "bindr"        "plyr"        
## [17] "stringr"      "MatrixModels" "munsell"      "gtable"      
## [21] "codetools"    "psych"        "evaluate"     "knitr"       
## [25] "SparseM"      "class"        "quantreg"     "pbkrtest"    
## [29] "parallel"     "Rcpp"         "backports"    "scales"      
## [33] "lme4"         "mnormt"       "digest"       "stringi"     
## [37] "bookdown"     "dplyr"        "grid"         "rprojroot"   
## [41] "tools"        "magrittr"     "lazyeval"     "tibble"      
## [45] "tidyr"        "car"          "pkgconfig"    "MASS"        
## [49] "assertthat"   "minqa"        "rmarkdown"    "iterators"   
## [53] "R6"           "nnet"         "nlme"         "compiler"
```
