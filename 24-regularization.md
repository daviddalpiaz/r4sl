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
## (Intercept) 159.796625075
## AtBat         0.102483884
## Hits          0.446840519
## HmRun         1.289060569
## Runs          0.702915318
## RBI           0.686866069
## Walks         0.925962429
## Years         2.714623469
## CAtBat        0.008746278
## CHits         0.034359576
## CHmRun        0.253594871
## CRuns         0.068874010
## CRBI          0.071334608
## CWalks        0.066114944
## LeagueN       5.396487460
## DivisionW   -29.096663826
## PutOuts       0.067805863
## Assists       0.009201998
## Errors       -0.235989099
## NewLeagueN    4.457548079
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
## (Intercept) 159.796625075
## AtBat         0.102483884
## Hits          0.446840519
## HmRun         1.289060569
## Runs          0.702915318
## RBI           0.686866069
## Walks         0.925962429
## Years         2.714623469
## CAtBat        0.008746278
## CHits         0.034359576
## CHmRun        0.253594871
## CRuns         0.068874010
## CRBI          0.071334608
## CWalks        0.066114944
## LeagueN       5.396487460
## DivisionW   -29.096663826
## PutOuts       0.067805863
## Assists       0.009201998
## Errors       -0.235989099
## NewLeagueN    4.457548079
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 906.8121
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 125141.2
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 450.9172 449.1847 448.5033 448.2332 447.9380 447.6154 447.2631
##  [8] 446.8784 446.4587 446.0009 445.5020 444.9585 444.3671 443.7238
## [15] 443.0247 442.2657 441.4426 440.5509 439.5861 438.5435 437.4184
## [22] 436.2062 434.9023 433.5023 432.0020 430.3977 428.6861 426.8644
## [29] 424.9307 422.8839 420.7239 418.4519 416.0702 413.5828 410.9950
## [36] 408.3140 405.5485 402.7087 399.8065 396.8554 393.8703 390.8671
## [43] 387.8623 384.8732 381.9170 379.0107 376.1706 373.4119 370.7485
## [50] 368.1922 365.7531 363.4396 361.2578 359.2115 357.3024 355.5303
## [57] 353.8936 352.3886 351.0109 349.7548 348.6139 347.5814 346.6503
## [64] 345.8131 345.0630 344.3976 343.8036 343.2764 342.8086 342.3971
## [71] 342.0390 341.7268 341.4487 341.2089 340.9972 340.8139 340.6516
## [78] 340.5039 340.3763 340.2547 340.1439 340.0363 339.9307 339.8244
## [85] 339.7168 339.6041 339.4840 339.3570 339.2225 339.0775 338.9219
## [92] 338.7576 338.5837 338.3999 338.2074 338.0051 337.7971 337.5817
## [99] 337.3626
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.3626
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 359.2115
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
##  [1] 449.6436 442.3096 432.6259 423.8990 415.6806 407.3209 398.9025
##  [8] 391.1110 384.3371 378.9044 374.4515 370.6835 367.5273 364.8804
## [15] 362.2557 359.5938 357.2282 355.1167 353.0105 351.3070 349.9240
## [22] 348.7711 347.8264 347.0502 346.4028 345.8781 345.4477 345.1866
## [29] 345.0281 344.9565 344.9136 344.8925 344.9130 345.0308 345.2494
## [36] 345.6030 346.2584 346.9522 347.3492 346.9504 345.8956 344.9304
## [43] 344.0914 343.2722 342.3536 341.6006 341.0414 340.6151 340.3473
## [50] 340.1565 340.0110 340.0441 340.2245 340.4700 340.7679 341.0875
## [57] 341.4449 341.7156 341.9791 342.2123 342.3854 342.4836 342.5615
## [64] 342.6347 342.7207 342.8370 342.9517 343.0731 343.1949 343.2810
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.011
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.5273
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
## 1  255.2820965 202179.4  37908.16  240087.5 164271.20     0
## 2  232.6035386 195637.8  38915.86  234553.6 156721.89     1
## 3  211.9396813 187165.2  37785.74  224950.9 149379.47     2
## 4  193.1115442 179690.4  36512.90  216203.3 143177.50     2
## 5  175.9560468 172790.4  35384.70  208175.1 137405.70     3
## 6  160.3245966 165910.3  34404.48  200314.8 131505.86     4
## 7  146.0818013 159123.2  33058.99  192182.2 126064.22     4
## 8  133.1042967 152967.8  31684.64  184652.5 121283.17     4
## 9  121.2796778 147715.0  30490.33  178205.3 117224.67     4
## 10 110.5055255 143568.6  29475.35  173043.9 114093.21     4
## 11 100.6885192 140213.9  28620.85  168834.8 111593.09     5
## 12  91.7436287 137406.3  27925.56  165331.8 109480.73     5
## 13  83.5933775 135076.3  27357.02  162433.3 107719.30     5
## 14  76.1671723 133137.7  26861.92  159999.6 106275.78     5
## 15  69.4006906 131229.2  26378.83  157608.0 104850.37     6
## 16  63.2353245 129307.7  25919.11  155226.8 103388.59     6
## 17  57.6176726 127612.0  25483.39  153095.4 102128.63     6
## 18  52.4990774 126107.8  25057.93  151165.8 101049.92     6
## 19  47.8352040 124616.4  24558.21  149174.6 100058.22     6
## 20  43.5856563 123416.6  24128.15  147544.8  99288.46     6
## 21  39.7136268 122446.8  23759.03  146205.8  98687.76     6
## 22  36.1855776 121641.3  23436.65  145077.9  98204.61     6
## 23  32.9709506 120983.2  23157.14  144140.4  97826.09     6
## 24  30.0419022 120443.9  22915.86  143359.7  97528.00     6
## 25  27.3730624 119994.9  22707.72  142702.6  97287.20     6
## 26  24.9413150 119631.7  22529.28  142161.0  97102.40     6
## 27  22.7255973 119334.1  22373.65  141707.8  96960.46     6
## 28  20.7067179 119153.8  22243.47  141397.2  96910.30     6
## 29  18.8671902 119044.4  22130.47  141174.8  96913.90     6
## 30  17.1910810 118995.0  22027.39  141022.4  96967.60     7
## 31  15.6638727 118965.4  21934.49  140899.9  97030.91     7
## 32  14.2723374 118950.9  21845.18  140796.0  97105.67     7
## 33  13.0044223 118965.0  21777.60  140742.6  97187.36     9
## 34  11.8491453 119046.3  21720.30  140766.6  97325.99     9
## 35  10.7964999 119197.1  21657.17  140854.3  97539.96     9
## 36   9.8373686 119441.4  21596.52  141037.9  97844.91     9
## 37   8.9634439 119894.9  21543.01  141437.9  98351.90     9
## 38   8.1671562 120375.8  21505.35  141881.2  98870.50    11
## 39   7.4416086 120651.5  21447.35  142098.8  99204.12    11
## 40   6.7805166 120374.5  21308.41  141683.0  99066.14    12
## 41   6.1781542 119643.8  21071.97  140715.7  98571.80    12
## 42   5.6293040 118976.9  20833.75  139810.7  98143.20    13
## 43   5.1292121 118398.9  20630.13  139029.1  97768.80    13
## 44   4.6735471 117835.8  20450.95  138286.7  97384.83    13
## 45   4.2583620 117206.0  20296.77  137502.8  96909.22    13
## 46   3.8800609 116691.0  20148.70  136839.7  96542.28    13
## 47   3.5353670 116309.3  19990.38  136299.6  96318.89    13
## 48   3.2212947 116018.6  19851.49  135870.1  96167.13    13
## 49   2.9351238 115836.3  19746.37  135582.7  96089.91    13
## 50   2.6743755 115706.4  19652.87  135359.3  96053.57    13
## 51   2.4367913 115607.5  19569.60  135177.1  96037.90    13
## 52   2.2203135 115630.0  19495.41  135125.4  96134.60    14
## 53   2.0230670 115752.7  19431.26  135183.9  96321.43    15
## 54   1.8433433 115919.8  19379.68  135299.5  96540.16    15
## 55   1.6795857 116122.8  19319.89  135442.7  96802.91    17
## 56   1.5303760 116340.7  19258.18  135598.8  97082.48    17
## 57   1.3944216 116584.6  19208.59  135793.2  97376.02    17
## 58   1.2705450 116769.5  19152.21  135921.7  97617.32    17
## 59   1.1576733 116949.7  19097.25  136046.9  97852.43    17
## 60   1.0548288 117109.3  19041.83  136151.1  98067.42    17
## 61   0.9611207 117227.8  18982.97  136210.7  98244.81    17
## 62   0.8757374 117295.0  18925.22  136220.2  98369.79    17
## 63   0.7979393 117348.4  18873.32  136221.7  98475.05    17
## 64   0.7270526 117398.6  18829.98  136228.5  98568.58    17
## 65   0.6624632 117457.4  18788.42  136245.9  98669.03    18
## 66   0.6036118 117537.2  18756.42  136293.6  98780.78    18
## 67   0.5499886 117615.8  18731.25  136347.1  98884.59    18
## 68   0.5011291 117699.2  18712.54  136411.7  98986.63    17
## 69   0.4566102 117782.7  18690.84  136473.6  99091.87    18
## 70   0.4160462 117841.8  18674.71  136516.6  99167.13    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   83.59338
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
