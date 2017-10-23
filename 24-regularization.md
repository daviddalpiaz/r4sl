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
## (Intercept) 240.682852262
## AtBat         0.083042199
## Hits          0.334990595
## HmRun         1.105720594
## Runs          0.542496738
## RBI           0.545363022
## Walks         0.698162048
## Years         2.316374820
## CAtBat        0.006988341
## CHits         0.026721778
## CHmRun        0.198876945
## CRuns         0.053594554
## CRBI          0.055409116
## CWalks        0.054405147
## LeagueN       2.463634059
## DivisionW   -18.860043802
## PutOuts       0.046088692
## Assists       0.006674057
## Errors       -0.112913895
## NewLeagueN    2.358350957
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
## (Intercept) 240.682852262
## AtBat         0.083042199
## Hits          0.334990595
## HmRun         1.105720594
## Runs          0.542496738
## RBI           0.545363022
## Walks         0.698162048
## Years         2.316374820
## CAtBat        0.006988341
## CHits         0.026721778
## CHmRun        0.198876945
## CRuns         0.053594554
## CRBI          0.055409116
## CWalks        0.054405147
## LeagueN       2.463634059
## DivisionW   -18.860043802
## PutOuts       0.046088692
## Assists       0.006674057
## Errors       -0.112913895
## NewLeagueN    2.358350957
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 375.1832
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 136525.7
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.4053 449.6649 448.9083 448.6449 448.3569 448.0422 447.6985
##  [8] 447.3233 446.9139 446.4674 445.9808 445.4508 444.8740 444.2467
## [15] 443.5651 442.8250 442.0224 441.1531 440.2125 439.1962 438.0996
## [22] 436.9182 435.6476 434.2835 432.8220 431.2593 429.5924 427.8187
## [29] 425.9362 423.9441 421.8424 419.6323 417.3163 414.8983 412.3837
## [36] 409.7796 407.0948 404.3392 401.5247 398.6647 395.7737 392.8673
## [43] 389.9620 387.0745 384.2217 381.4201 378.6857 376.0332 373.4759
## [50] 371.0250 368.6906 366.4802 364.3995 362.4516 360.6379 358.9579
## [57] 357.4092 355.9881 354.6894 353.5074 352.4354 351.4663 350.5928
## [64] 349.8076 349.1072 348.4815 347.9223 347.4249 346.9837 346.5946
## [71] 346.2515 345.9503 345.6815 345.4472 345.2405 345.0614 344.8992
## [78] 344.7564 344.6267 344.5103 344.4032 344.3041 344.2089 344.1182
## [85] 344.0278 343.9374 343.8462 343.7533 343.6563 343.5566 343.4523
## [92] 343.3440 343.2314 343.1150 342.9942 342.8699 342.7428 342.6130
## [99] 342.4805
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.4805
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 376.0332
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
##                         1
## (Intercept)  134.48030406
## AtBat         -1.67572220
## Hits           5.94122316
## HmRun          0.04746835
## Runs           .         
## RBI            .         
## Walks          4.95676182
## Years        -10.26657309
## CAtBat         .         
## CHits          .         
## CHmRun         0.56236426
## CRuns          0.70135135
## CRBI           0.38727139
## CWalks        -0.58111548
## LeagueN       32.92255640
## DivisionW   -119.37941356
## PutOuts        0.27580087
## Assists        0.19782326
## Errors        -2.26242857
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 180.1579
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
##  [1] 450.3177 442.0982 432.5157 424.1321 416.6425 408.4227 400.6286
##  [8] 393.5822 387.4319 382.2270 377.9173 374.1717 370.7683 367.7501
## [15] 364.8885 362.0639 359.3708 356.9579 354.9041 353.2098 351.8102
## [22] 350.6514 349.6965 348.9313 348.3925 347.9784 347.6483 347.4184
## [29] 347.2879 347.2158 347.1735 347.1730 347.2335 347.5513 348.0212
## [36] 348.5619 349.2291 349.7790 349.9052 349.6730 349.0408 348.3266
## [43] 347.5218 346.6865 345.9764 345.3784 344.9247 344.5888 344.3334
## [50] 344.1814 344.1005 344.0531 344.0659 344.2104 344.2920 344.3619
## [57] 344.4070 344.5150 344.6530 344.7831 344.8903 344.9790 345.0625
## [64] 345.1677 345.2987 345.4276 345.5599 345.6845 345.8297 345.9547
## [71] 346.1025 346.2063 346.3539
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.0531
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.7683
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
## 1  255.2820965 202786.0  30412.34  233198.3 172373.66     0
## 2  232.6035386 195450.8  30619.37  226070.2 164831.42     1
## 3  211.9396813 187069.8  29439.46  216509.3 157630.38     2
## 4  193.1115442 179888.0  27955.78  207843.8 151932.24     2
## 5  175.9560468 173591.0  26771.51  200362.5 146819.46     3
## 6  160.3245966 166809.1  25680.97  192490.0 141128.10     4
## 7  146.0818013 160503.3  24718.26  185221.6 135785.04     4
## 8  133.1042967 154907.0  23962.24  178869.2 130944.74     4
## 9  121.2796778 150103.5  23400.68  173504.1 126702.78     4
## 10 110.5055255 146097.5  23023.81  169121.3 123073.68     4
## 11 100.6885192 142821.5  22802.38  165623.9 120019.14     5
## 12  91.7436287 140004.5  22695.14  162699.6 117309.31     5
## 13  83.5933775 137469.1  22658.43  160127.5 114810.68     5
## 14  76.1671723 135240.1  22677.90  157918.0 112562.21     5
## 15  69.4006906 133143.6  22741.06  155884.7 110402.55     6
## 16  63.2353245 131090.3  22770.23  153860.5 108320.06     6
## 17  57.6176726 129147.4  22726.09  151873.4 106421.26     6
## 18  52.4990774 127418.9  22643.51  150062.4 104775.42     6
## 19  47.8352040 125957.0  22563.68  148520.6 103393.27     6
## 20  43.5856563 124757.1  22505.84  147263.0 102251.30     6
## 21  39.7136268 123770.4  22465.70  146236.1 101304.74     6
## 22  36.1855776 122956.4  22439.89  145396.3 100516.52     6
## 23  32.9709506 122287.7  22424.81  144712.5  99862.85     6
## 24  30.0419022 121753.1  22420.01  144173.1  99333.05     6
## 25  27.3730624 121377.4  22433.16  143810.5  98944.19     6
## 26  24.9413150 121089.0  22452.92  143541.9  98636.04     6
## 27  22.7255973 120859.3  22474.48  143333.8  98384.84     6
## 28  20.7067179 120699.5  22499.13  143198.7  98200.41     6
## 29  18.8671902 120608.9  22520.65  143129.5  98088.21     6
## 30  17.1910810 120558.8  22536.71  143095.6  98022.14     7
## 31  15.6638727 120529.4  22547.97  143077.4  97981.45     7
## 32  14.2723374 120529.1  22560.55  143089.6  97968.51     7
## 33  13.0044223 120571.1  22582.61  143153.7  97988.47     9
## 34  11.8491453 120791.9  22615.69  143407.6  98176.19     9
## 35  10.7964999 121118.8  22635.62  143754.4  98483.13     9
## 36   9.8373686 121495.4  22646.64  144142.1  98848.77     9
## 37   8.9634439 121961.0  22640.55  144601.5  99320.42     9
## 38   8.1671562 122345.3  22566.29  144911.6  99779.03    11
## 39   7.4416086 122433.7  22445.21  144878.9  99988.44    11
## 40   6.7805166 122271.2  22285.82  144557.0  99985.36    12
## 41   6.1781542 121829.5  22053.21  143882.7  99776.28    12
## 42   5.6293040 121331.5  21795.68  143127.1  99535.77    13
## 43   5.1292121 120771.4  21532.30  142303.7  99239.08    13
## 44   4.6735471 120191.5  21229.99  141421.5  98961.55    13
## 45   4.2583620 119699.6  20913.74  140613.4  98785.90    13
## 46   3.8800609 119286.2  20610.55  139896.8  98675.66    13
## 47   3.5353670 118973.1  20313.55  139286.6  98659.51    13
## 48   3.2212947 118741.5  20036.44  138777.9  98705.02    13
## 49   2.9351238 118565.5  19792.12  138357.6  98773.38    13
## 50   2.6743755 118460.8  19576.84  138037.7  98883.97    13
## 51   2.4367913 118405.2  19385.03  137790.2  99020.12    13
## 52   2.2203135 118372.5  19211.27  137583.8  99161.24    14
## 53   2.0230670 118381.4  19049.79  137431.2  99331.58    15
## 54   1.8433433 118480.8  18908.26  137389.1  99572.53    15
## 55   1.6795857 118537.0  18789.18  137326.2  99747.80    17
## 56   1.5303760 118585.1  18683.20  137268.3  99901.94    17
## 57   1.3944216 118616.2  18586.43  137202.6 100029.75    17
## 58   1.2705450 118690.6  18501.03  137191.6 100189.58    17
## 59   1.1576733 118785.7  18421.27  137207.0 100364.43    17
## 60   1.0548288 118875.4  18350.27  137225.6 100525.11    17
## 61   0.9611207 118949.3  18282.63  137231.9 100666.69    17
## 62   0.8757374 119010.5  18221.14  137231.7 100789.38    17
## 63   0.7979393 119068.1  18166.96  137235.1 100901.14    17
## 64   0.7270526 119140.8  18121.87  137262.6 101018.90    17
## 65   0.6624632 119231.2  18084.37  137315.6 101146.85    18
## 66   0.6036118 119320.2  18055.17  137375.4 101265.04    18
## 67   0.5499886 119411.7  18029.16  137440.8 101382.51    18
## 68   0.5011291 119497.7  18003.95  137501.7 101493.79    17
## 69   0.4566102 119598.2  17979.78  137577.9 101618.38    18
## 70   0.4160462 119684.6  17961.38  137646.0 101723.27    18
## 71   0.3790858 119787.0  17937.61  137724.6 101849.34    18
## 72   0.3454089 119858.8  17924.64  137783.5 101934.19    18
## 73   0.3147237 119961.0  17906.40  137867.4 102054.65    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.220313   83.59338
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

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.4.2 and the following packages:

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
## [61] "sfsmisc"      "parallel"     "survival"     "yaml"        
## [65] "colorspace"   "knitr"        "bindr"
```
