# Shrinkage Methods

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

![](15-shrink_files/figure-latex/unnamed-chunk-7-1.pdf)<!-- --> 

```r
plot(fit_ridge, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/unnamed-chunk-7-2.pdf)<!-- --> 

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

![](15-shrink_files/figure-latex/unnamed-chunk-8-1.pdf)<!-- --> 

The `cv.glmnet()` function returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are smaller than the full linear regression. (As we would expect.)


```r
coef(fit_ridge_cv)
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 254.518230141
## AtBat         0.079408204
## Hits          0.317375220
## HmRun         1.065097243
## Runs          0.515835607
## RBI           0.520504723
## Walks         0.661891621
## Years         2.231379426
## CAtBat        0.006679258
## CHits         0.025455999
## CHmRun        0.189661478
## CRuns         0.051057906
## CRBI          0.052776153
## CWalks        0.052170266
## LeagueN       2.114989228
## DivisionW   -17.479743519
## PutOuts       0.043039515
## Assists       0.006296277
## Errors       -0.099487300
## NewLeagueN    2.085946064
```

```r
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  10.275516216
## AtBat         0.008088527
## Hits          1.075046175
## HmRun        -0.046053688
## Runs          1.128360908
## RBI           0.868722584
## Walks         1.876642475
## Years        -0.425145113
## CAtBat        0.010952415
## CHits         0.068113575
## CHmRun        0.469845085
## CRuns         0.135302716
## CRBI          0.144283368
## CWalks        0.017175670
## LeagueN      29.161665212
## DivisionW   -95.881063089
## PutOuts       0.200221069
## Assists       0.048671621
## Errors       -1.988817468
## NewLeagueN    6.058926651
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 10091.44
```

```r
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 254.518230141
## AtBat         0.079408204
## Hits          0.317375220
## HmRun         1.065097243
## Runs          0.515835607
## RBI           0.520504723
## Walks         0.661891621
## Years         2.231379426
## CAtBat        0.006679258
## CHits         0.025455999
## CHmRun        0.189661478
## CRuns         0.051057906
## CRBI          0.052776153
## CWalks        0.052170266
## LeagueN       2.114989228
## DivisionW   -17.479743519
## PutOuts       0.043039515
## Assists       0.006296277
## Errors       -0.099487300
## NewLeagueN    2.085946064
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 321.618
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 138732.7
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.9532 450.1166 449.4786 449.2129 448.9225 448.6052 448.2586
##  [8] 447.8802 447.4673 447.0171 446.5263 445.9918 445.4100 444.7773
## [15] 444.0898 443.3433 442.5338 441.6568 440.7080 439.6827 438.5762
## [22] 437.3842 436.1020 434.7254 433.2503 431.6729 429.9901 428.1991
## [29] 426.2981 424.2860 422.1627 419.9293 417.5882 415.1433 412.5999
## [36] 409.9649 407.2472 404.4562 401.6042 398.7042 395.7708 392.8196
## [43] 389.8671 386.9299 384.0253 381.1697 378.3794 375.6693 373.0533
## [50] 370.5429 368.1487 365.8785 363.7381 361.7323 359.8628 358.1296
## [57] 356.5312 355.0646 353.7255 352.5087 351.4082 350.4178 349.5306
## [64] 348.7382 348.0341 347.4194 346.8800 346.4117 346.0057 345.6624
## [71] 345.3799 345.1449 344.9480 344.7984 344.6874 344.6026 344.5502
## [78] 344.5183 344.5131 344.5203 344.5472 344.5830 344.6210 344.6733
## [85] 344.7270 344.7804 344.8258 344.8766 344.9197 344.9490 344.9783
## [92] 344.9825 345.0089 344.9846 344.9974 344.9480 344.9372 344.8738
## [99] 344.8473
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.5131
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 378.3794
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

![](15-shrink_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

```r
plot(fit_lasso, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/unnamed-chunk-10-2.pdf)<!-- --> 

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

![](15-shrink_files/figure-latex/unnamed-chunk-11-1.pdf)<!-- --> 

`cv.glmnet()` returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are again smaller than the full linear regression. (As we would expect.) Some coefficients are 0.


```r
coef(fit_lasso_cv)
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 167.91202818
## AtBat         .         
## Hits          1.29269756
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.39817511
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.14167760
## CRBI          0.32192558
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.04675463
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
## (Intercept)  117.5258439
## AtBat         -1.4742901
## Hits           5.4994256
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.5991651
## Years         -9.1918308
## CAtBat         .        
## CHits          .        
## CHmRun         0.4806743
## CRuns          0.6354799
## CRBI           0.3956153
## CWalks        -0.4993240
## LeagueN       31.6238174
## DivisionW   -119.2516409
## PutOuts        0.2704287
## Assists        0.1594997
## Errors        -1.9426357
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 176.0238
```

```r
coef(fit_lasso_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 167.91202818
## AtBat         .         
## Hits          1.29269756
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.39817511
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.14167760
## CRBI          0.32192558
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.04675463
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 3.20123
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 123931.3
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 451.0624 441.0459 431.1986 423.3527 416.3081 407.7212 399.4630
##  [8] 392.2630 386.0643 380.8778 376.5654 372.8120 369.1883 365.9608
## [15] 362.8611 359.7878 357.0536 354.7686 352.8948 351.3525 350.0948
## [22] 349.1379 348.5081 348.0718 347.7617 347.6382 347.6162 347.6589
## [29] 347.7340 347.8355 347.9665 348.1257 348.2265 348.3577 348.5563
## [36] 348.9707 349.4458 349.7699 349.3706 348.6777 347.8811 347.1212
## [43] 346.4159 345.9192 345.3744 344.9593 344.6002 344.3251 344.1429
## [50] 344.2710 344.4450 344.6597 344.8639 345.0899 345.4070 345.7092
## [57] 345.9087 346.2123 346.5400 346.9297 347.2658 347.5352 347.7144
## [64] 347.9526 348.1297 348.3406 348.5128 348.7363 349.0247 349.2011
## [71] 349.2869
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.1429
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.1883
```

## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
#lassoModCV
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high  conf.low nzero
## 1  255.2820965 203457.3  17223.14  220680.4 186234.17     0
## 2  232.6035386 194521.4  17844.56  212366.0 176676.88     1
## 3  211.9396813 185932.2  17500.96  203433.2 168431.24     2
## 4  193.1115442 179227.5  17228.26  196455.7 161999.24     2
## 5  175.9560468 173312.5  17079.07  190391.5 156233.41     3
## 6  160.3245966 166236.6  16944.00  183180.6 149292.55     4
## 7  146.0818013 159570.7  16658.00  176228.7 142912.70     4
## 8  133.1042967 153870.3  16462.07  170332.4 137408.21     4
## 9  121.2796778 149045.7  16388.11  165433.8 132657.55     4
## 10 110.5055255 145067.9  16379.29  161447.2 128688.65     4
## 11 100.6885192 141801.5  16440.96  158242.4 125360.53     5
## 12  91.7436287 138988.8  16576.91  155565.7 122411.89     5
## 13  83.5933775 136300.0  16738.95  153039.0 119561.05     5
## 14  76.1671723 133927.3  16890.05  150817.4 117037.27     5
## 15  69.4006906 131668.2  17108.78  148777.0 114559.43     6
## 16  63.2353245 129447.2  17247.15  146694.4 112200.09     6
## 17  57.6176726 127487.2  17329.03  144816.3 110158.21     6
## 18  52.4990774 125860.8  17430.28  143291.0 108430.48     6
## 19  47.8352040 124534.7  17546.96  142081.7 106987.77     6
## 20  43.5856563 123448.6  17674.55  141123.1 105774.04     6
## 21  39.7136268 122566.4  17809.70  140376.1 104756.66     6
## 22  36.1855776 121897.3  17959.64  139856.9 103937.66     6
## 23  32.9709506 121457.9  18133.92  139591.8 103323.99     6
## 24  30.0419022 121154.0  18310.69  139464.7 102843.31     6
## 25  27.3730624 120938.2  18482.37  139420.6 102455.81     6
## 26  24.9413150 120852.3  18654.68  139507.0 102197.66     6
## 27  22.7255973 120837.0  18823.74  139660.8 102013.29     6
## 28  20.7067179 120866.7  18986.87  139853.6 101879.82     6
## 29  18.8671902 120918.9  19138.24  140057.2 101780.69     6
## 30  17.1910810 120989.5  19277.69  140267.2 101711.82     7
## 31  15.6638727 121080.7  19406.38  140487.1 101674.30     7
## 32  14.2723374 121191.5  19527.04  140718.5 101664.47     7
## 33  13.0044223 121261.7  19647.05  140908.8 101614.68     9
## 34  11.8491453 121353.1  19763.52  141116.6 101589.55     9
## 35  10.7964999 121491.5  19887.80  141379.3 101603.70     9
## 36   9.8373686 121780.6  20008.15  141788.7 101772.42     9
## 37   8.9634439 122112.4  20088.01  142200.4 102024.38     9
## 38   8.1671562 122339.0  20154.54  142493.5 102184.41    11
## 39   7.4416086 122059.8  20078.59  142138.4 101981.20    11
## 40   6.7805166 121576.1  19877.69  141453.8 101698.42    12
## 41   6.1781542 121021.3  19671.77  140693.0 101349.50    12
## 42   5.6293040 120493.1  19500.99  139994.1 100992.14    13
## 43   5.1292121 120004.0  19358.93  139362.9 100645.08    13
## 44   4.6735471 119660.1  19238.66  138898.7 100421.42    13
## 45   4.2583620 119283.5  19134.37  138417.9 100149.11    13
## 46   3.8800609 118996.9  19015.41  138012.3  99981.48    13
## 47   3.5353670 118749.3  18860.03  137609.4  99889.30    13
## 48   3.2212947 118559.8  18700.93  137260.7  99858.87    13
## 49   2.9351238 118434.3  18584.78  137019.1  99849.56    13
## 50   2.6743755 118522.5  18468.57  136991.1 100053.96    13
## 51   2.4367913 118642.4  18362.52  137004.9 100279.86    13
## 52   2.2203135 118790.3  18258.03  137048.3 100532.25    14
## 53   2.0230670 118931.1  18152.46  137083.6 100778.65    15
## 54   1.8433433 119087.1  18042.77  137129.8 101044.29    15
## 55   1.6795857 119306.0  17930.54  137236.6 101375.48    17
## 56   1.5303760 119514.9  17802.75  137317.6 101712.13    17
## 57   1.3944216 119652.8  17657.78  137310.6 101995.03    17
## 58   1.2705450 119862.9  17535.62  137398.5 102327.31    17
## 59   1.1576733 120090.0  17429.62  137519.6 102660.35    17
## 60   1.0548288 120360.2  17342.14  137702.3 103018.05    17
## 61   0.9611207 120593.5  17252.62  137846.2 103340.92    17
## 62   0.8757374 120780.7  17175.46  137956.2 103605.25    17
## 63   0.7979393 120905.3  17107.46  138012.8 103797.86    17
## 64   0.7270526 121071.0  17059.72  138130.7 104011.29    17
## 65   0.6624632 121194.3  17010.43  138204.7 104183.87    18
## 66   0.6036118 121341.2  16974.87  138316.0 104366.29    18
## 67   0.5499886 121461.2  16938.75  138400.0 104522.45    18
## 68   0.5011291 121617.0  16911.14  138528.2 104705.89    17
## 69   0.4566102 121818.2  16892.64  138710.9 104925.60    18
## 70   0.4160462 121941.4  16866.40  138807.8 105075.02    18
## 71   0.3790858 122001.3  16839.40  138840.7 105161.93    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.935124   83.59338
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

![](15-shrink_files/figure-latex/unnamed-chunk-16-1.pdf)<!-- --> 


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

![](15-shrink_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 


```r
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```

![](15-shrink_files/figure-latex/unnamed-chunk-21-1.pdf)<!-- --> 

We can extracting the two relevant $\lambda$ values.


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
fit_lasso = train(
  x = X,
  y = y,
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

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.3.2 and the following packages:

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
##  [1] "Rcpp"         "compiler"     "nloptr"       "plyr"        
##  [5] "class"        "iterators"    "tools"        "lme4"        
##  [9] "digest"       "evaluate"     "tibble"       "nlme"        
## [13] "gtable"       "mgcv"         "psych"        "DBI"         
## [17] "yaml"         "parallel"     "SparseM"      "e1071"       
## [21] "dplyr"        "stringr"      "knitr"        "MatrixModels"
## [25] "stats4"       "nnet"         "rprojroot"    "grid"        
## [29] "R6"           "foreign"      "rmarkdown"    "bookdown"    
## [33] "minqa"        "car"          "reshape2"     "tidyr"       
## [37] "magrittr"     "splines"      "MASS"         "ModelMetrics"
## [41] "backports"    "scales"       "codetools"    "htmltools"   
## [45] "pbkrtest"     "assertthat"   "mnormt"       "colorspace"  
## [49] "quantreg"     "stringi"      "lazyeval"     "munsell"
```
