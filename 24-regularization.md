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
##  [1] 452.6613 451.3631 450.6299 450.0399 449.7554 449.4445 449.1049
##  [8] 448.7342 448.3297 447.8886 447.4078 446.8841 446.3142 445.6944
## [15] 445.0208 444.2896 443.4966 442.6376 441.7082 440.7039 439.6203
## [22] 438.4528 437.1971 435.8490 434.4045 432.8600 431.2122 429.4588
## [29] 427.5976 425.6279 423.5494 421.3634 419.0721 416.6794 414.1905
## [36] 411.6122 408.9530 406.2225 403.4325 400.5957 397.7265 394.8400
## [43] 391.9523 389.0799 386.2392 383.4466 380.7178 378.0674 375.5088
## [50] 373.0533 370.7108 368.4894 366.3947 364.4305 362.5992 360.9005
## [57] 359.3329 357.8931 356.5770 355.3793 354.2940 353.3146 352.4346
## [64] 351.6454 350.9443 350.3225 349.7725 349.2908 348.8679 348.5012
## [71] 348.1853 347.9083 347.6739 347.4758 347.3042 347.1618 347.0372
## [78] 346.9313 346.8387 346.7577 346.6814 346.6120 346.5407 346.4676
## [85] 346.3909 346.3094 346.2150 346.1176 346.0050 345.8867 345.7506
## [92] 345.6083 345.4475 345.2815 345.1000 344.9126 344.7144 344.5084
## [99] 344.3007
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.3007
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.7108
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
##  [1] 450.8135 442.5532 433.4290 424.6255 416.1697 407.4882 399.6543
##  [8] 392.8753 386.9531 381.8083 377.2600 373.3153 369.8228 366.6715
## [15] 363.8728 361.2672 358.6836 356.2146 354.1486 352.4343 351.0087
## [22] 349.8201 348.8414 348.0345 347.4264 347.0700 346.9090 346.8217
## [29] 346.7813 346.7847 346.8383 346.9129 347.0065 347.1306 347.3936
## [36] 347.9711 348.7852 349.6190 349.9440 349.8199 349.2357 348.6191
## [43] 347.9016 347.1184 346.3227 345.6955 345.1821 344.7866 344.4994
## [50] 344.3416 344.3082 344.2809 344.3284 344.4638 344.4550 344.3832
## [57] 344.4034 344.4351 344.4899 344.5530 344.6485 344.7719 344.9589
## [64] 345.1635 345.3333 345.5886 345.8171 346.0642 346.2998 346.5144
## [71] 346.7232 346.8917 347.0714 347.2212
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.2809
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.8228
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
## 1  255.2820965 203232.8  36129.45  239362.2 167103.32     0
## 2  232.6035386 195853.3  36424.09  232277.4 159429.21     1
## 3  211.9396813 187860.7  35789.07  223649.7 152071.59     2
## 4  193.1115442 180306.8  34532.87  214839.7 145773.96     2
## 5  175.9560468 173197.2  33430.27  206627.5 139766.94     3
## 6  160.3245966 166046.6  32417.19  198463.8 133629.40     4
## 7  146.0818013 159723.6  31429.03  191152.6 128294.54     4
## 8  133.1042967 154351.0  30502.15  184853.2 123848.88     4
## 9  121.2796778 149732.7  29628.33  179361.1 120104.39     4
## 10 110.5055255 145777.6  28850.84  174628.4 116926.74     4
## 11 100.6885192 142325.1  28170.61  170495.7 114154.47     5
## 12  91.7436287 139364.3  27594.16  166958.5 111770.15     5
## 13  83.5933775 136768.9  27073.97  163842.8 109694.90     5
## 14  76.1671723 134448.0  26598.81  161046.8 107849.17     5
## 15  69.4006906 132403.4  26186.85  158590.3 106216.58     6
## 16  63.2353245 130514.0  25806.06  156320.0 104707.90     6
## 17  57.6176726 128653.9  25413.09  154067.0 103240.83     6
## 18  52.4990774 126888.8  25032.17  151921.0 101856.64     6
## 19  47.8352040 125421.3  24706.45  150127.7 100714.82     6
## 20  43.5856563 124209.9  24428.30  148638.2  99781.61     6
## 21  39.7136268 123207.1  24191.32  147398.4  99015.76     6
## 22  36.1855776 122374.1  23989.39  146363.5  98384.72     6
## 23  32.9709506 121690.3  23816.15  145506.5  97874.18     6
## 24  30.0419022 121128.0  23667.41  144795.4  97460.58     6
## 25  27.3730624 120705.1  23546.86  144251.9  97158.22     6
## 26  24.9413150 120457.6  23454.13  143911.7  97003.45     6
## 27  22.7255973 120345.8  23374.85  143720.7  96970.99     6
## 28  20.7067179 120285.3  23305.22  143590.5  96980.09     6
## 29  18.8671902 120257.3  23244.44  143501.7  97012.86     6
## 30  17.1910810 120259.6  23192.17  143451.8  97067.45     7
## 31  15.6638727 120296.8  23148.18  143445.0  97148.65     7
## 32  14.2723374 120348.6  23123.58  143472.2  97224.99     7
## 33  13.0044223 120413.5  23122.41  143535.9  97291.12     9
## 34  11.8491453 120499.6  23125.12  143624.7  97374.51     9
## 35  10.7964999 120682.3  23125.81  143808.1  97556.52     9
## 36   9.8373686 121083.9  23116.57  144200.5  97967.35     9
## 37   8.9634439 121651.1  23089.18  144740.3  98561.93     9
## 38   8.1671562 122233.5  23045.24  145278.7  99188.21    11
## 39   7.4416086 122460.8  23013.57  145474.4  99447.26    11
## 40   6.7805166 122373.9  22912.92  145286.9  99461.03    12
## 41   6.1781542 121965.6  22644.02  144609.6  99321.54    12
## 42   5.6293040 121535.3  22365.60  143900.9  99169.65    13
## 43   5.1292121 121035.5  22129.20  143164.7  98906.32    13
## 44   4.6735471 120491.2  21897.82  142389.0  98593.37    13
## 45   4.2583620 119939.4  21681.53  141620.9  98257.87    13
## 46   3.8800609 119505.4  21487.46  140992.8  98017.91    13
## 47   3.5353670 119150.7  21311.58  140462.3  97839.12    13
## 48   3.2212947 118877.8  21145.48  140023.3  97732.35    13
## 49   2.9351238 118679.9  20977.21  139657.1  97702.66    13
## 50   2.6743755 118571.1  20841.71  139412.9  97729.44    13
## 51   2.4367913 118548.1  20760.95  139309.1  97787.18    13
## 52   2.2203135 118529.4  20667.00  139196.4  97862.37    14
## 53   2.0230670 118562.0  20570.51  139132.5  97991.52    15
## 54   1.8433433 118655.3  20459.08  139114.4  98196.22    15
## 55   1.6795857 118649.2  20318.55  138967.8  98330.68    17
## 56   1.5303760 118599.8  20207.44  138807.2  98392.35    17
## 57   1.3944216 118613.7  20126.65  138740.3  98487.04    17
## 58   1.2705450 118635.6  20067.75  138703.3  98567.82    17
## 59   1.1576733 118673.3  20021.52  138694.8  98651.76    17
## 60   1.0548288 118716.8  19995.43  138712.2  98721.34    17
## 61   0.9611207 118782.6  20009.96  138792.5  98772.62    17
## 62   0.8757374 118867.6  20032.57  138900.2  98835.06    17
## 63   0.7979393 118996.6  20052.42  139049.0  98944.20    17
## 64   0.7270526 119137.8  20078.10  139215.9  99059.72    17
## 65   0.6624632 119255.1  20101.27  139356.3  99153.80    18
## 66   0.6036118 119431.5  20124.12  139555.6  99307.37    18
## 67   0.5499886 119589.5  20148.64  139738.1  99440.82    18
## 68   0.5011291 119760.4  20170.89  139931.3  99589.52    17
## 69   0.4566102 119923.6  20190.08  140113.6  99733.48    18
## 70   0.4160462 120072.2  20211.34  140283.6  99860.90    18
## 71   0.3790858 120217.0  20228.63  140445.6  99988.37    18
## 72   0.3454089 120333.9  20246.53  140580.4 100087.35    18
## 73   0.3147237 120458.6  20262.61  140721.2 100195.97    18
## 74   0.2867645 120562.6  20280.32  140842.9 100282.26    18
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
