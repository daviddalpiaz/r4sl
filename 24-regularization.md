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
##  [1] 453.6654 451.8201 451.1508 450.8820 450.5882 450.2672 449.9165
##  [8] 449.5337 449.1160 448.6605 448.1640 447.6232 447.0345 446.3944
## [15] 445.6988 444.9434 444.1243 443.2370 442.2769 441.2394 440.1199
## [22] 438.9137 437.6163 436.2233 434.7306 433.1345 431.4317 429.6195
## [29] 427.6960 425.6601 423.5119 421.2524 418.8843 416.4114 413.8393
## [36] 411.1751 408.4277 405.6072 402.7258 399.7972 396.8362 393.8589
## [43] 390.8820 387.9230 384.9991 382.1275 379.3246 376.6058 373.9850
## [50] 371.4743 369.0836 366.8205 364.6921 362.7017 360.8510 359.1395
## [57] 357.5651 356.1243 354.8122 353.6227 352.5493 351.5848 350.7218
## [64] 349.9508 349.2677 348.6663 348.1379 347.6760 347.2713 346.9256
## [71] 346.6277 346.3720 346.1522 345.9669 345.8086 345.6784 345.5664
## [78] 345.4651 345.3825 345.3061 345.2361 345.1719 345.0969 345.0292
## [85] 344.9595 344.8719 344.7857 344.6787 344.5818 344.4538 344.3268
## [92] 344.1847 344.0308 343.8676 343.6995 343.5135 343.3290 343.1306
## [99] 342.9337
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.9337
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 379.3246
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
## (Intercept) 127.95694754
## AtBat         .         
## Hits          1.42342566
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.58214111
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.16027975
## CRBI          0.33667715
## CWalks        .         
## LeagueN       .         
## DivisionW    -8.06171262
## PutOuts       0.08393604
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
## (Intercept) 127.95694754
## AtBat         .         
## Hits          1.42342566
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.58214111
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.16027975
## CRBI          0.33667715
## CWalks        .         
## LeagueN       .         
## DivisionW    -8.06171262
## PutOuts       0.08393604
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 11.64817
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 118581.5
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 450.3834 440.6440 430.3330 421.4177 412.9404 403.7149 394.6780
##  [8] 386.7149 379.9970 374.5681 370.0617 366.1179 362.6260 359.4646
## [15] 356.5536 353.7351 351.0864 348.5924 346.5384 344.8374 343.4276
## [22] 342.2663 341.3259 340.5568 339.9368 339.4231 339.0254 338.7462
## [29] 338.5619 338.4555 338.3834 338.3476 338.3490 338.3852 338.5237
## [36] 338.8146 339.0990 339.2405 339.0915 338.7814 338.3132 337.6608
## [43] 336.7480 335.9978 335.3557 334.8338 334.4771 334.2066 334.0420
## [50] 334.0606 334.1974 334.3832 334.6276 334.9874 335.4371 335.8561
## [57] 336.1733 336.4858 336.8297 337.1574 337.4745 337.7624 338.0231
## [64] 338.2810 338.5135 338.7287 338.9216 339.0944 339.2162 339.3294
## [71] 339.4521 339.5527 339.6585
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 334.042
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 356.5536
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
## 1  255.2820965 202845.2  26755.68  229600.9 176089.50     0
## 2  232.6035386 194167.1  26585.44  220752.6 167581.70     1
## 3  211.9396813 185186.5  25800.45  210987.0 159386.08     2
## 4  193.1115442 177592.9  25094.70  202687.6 152498.19     2
## 5  175.9560468 170519.8  24416.38  194936.2 146103.44     3
## 6  160.3245966 162985.7  23777.94  186763.7 139207.78     4
## 7  146.0818013 155770.8  23220.94  178991.7 132549.82     4
## 8  133.1042967 149548.4  22755.30  172303.7 126793.15     4
## 9  121.2796778 144397.8  22359.24  166757.0 122038.51     4
## 10 110.5055255 140301.2  22033.21  162334.4 118268.01     4
## 11 100.6885192 136945.6  21764.43  158710.1 115181.20     5
## 12  91.7436287 134042.3  21496.44  155538.7 112545.85     5
## 13  83.5933775 131497.6  21231.62  152729.2 110266.00     5
## 14  76.1671723 129214.8  20998.32  150213.1 108216.49     5
## 15  69.4006906 127130.4  20788.84  147919.3 106341.61     6
## 16  63.2353245 125128.5  20600.95  145729.5 104527.59     6
## 17  57.6176726 123261.7  20374.89  143636.6 102886.77     6
## 18  52.4990774 121516.6  20085.84  141602.5 101430.79     6
## 19  47.8352040 120088.9  19841.54  139930.4 100247.32     6
## 20  43.5856563 118912.8  19632.59  138545.4  99280.23     6
## 21  39.7136268 117942.5  19454.10  137396.6  98488.39     6
## 22  36.1855776 117146.2  19302.50  136448.7  97843.69     6
## 23  32.9709506 116503.4  19180.55  135683.9  97322.81     6
## 24  30.0419022 115978.9  19074.43  135053.4  96904.48     6
## 25  27.3730624 115557.0  18985.87  134542.9  96571.18     6
## 26  24.9413150 115208.0  18908.81  134116.8  96299.21     6
## 27  22.7255973 114938.2  18843.86  133782.1  96094.34     6
## 28  20.7067179 114749.0  18792.46  133541.4  95956.51     6
## 29  18.8671902 114624.1  18749.75  133373.9  95874.38     6
## 30  17.1910810 114552.1  18709.09  133261.2  95843.05     7
## 31  15.6638727 114503.3  18675.03  133178.3  95828.28     7
## 32  14.2723374 114479.1  18645.29  133124.4  95833.78     7
## 33  13.0044223 114480.1  18620.86  133100.9  95859.19     9
## 34  11.8491453 114504.6  18594.53  133099.1  95910.03     9
## 35  10.7964999 114598.3  18554.09  133152.4  96044.20     9
## 36   9.8373686 114795.3  18486.35  133281.7  96309.00     9
## 37   8.9634439 114988.1  18433.66  133421.8  96554.47     9
## 38   8.1671562 115084.1  18396.70  133480.8  96687.40    11
## 39   7.4416086 114983.1  18369.89  133352.9  96613.16    11
## 40   6.7805166 114772.9  18338.89  133111.8  96433.97    12
## 41   6.1781542 114455.8  18195.68  132651.5  96260.13    12
## 42   5.6293040 114014.8  17918.60  131933.4  96096.21    13
## 43   5.1292121 113399.2  17510.57  130909.8  95888.67    13
## 44   4.6735471 112894.5  17156.46  130051.0  95738.08    13
## 45   4.2583620 112463.4  16870.98  129334.4  95592.45    13
## 46   3.8800609 112113.7  16621.97  128735.7  95491.73    13
## 47   3.5353670 111874.9  16395.63  128270.6  95479.30    13
## 48   3.2212947 111694.0  16191.55  127885.6  95502.49    13
## 49   2.9351238 111584.1  16008.87  127592.9  95575.18    13
## 50   2.6743755 111596.5  15846.08  127442.6  95750.41    13
## 51   2.4367913 111687.9  15701.17  127389.1  95986.74    13
## 52   2.2203135 111812.1  15559.67  127371.8  96252.46    14
## 53   2.0230670 111975.6  15411.53  127387.2  96564.12    15
## 54   1.8433433 112216.5  15245.32  127461.9  96971.22    15
## 55   1.6795857 112518.1  15103.00  127621.1  97415.07    17
## 56   1.5303760 112799.3  14970.74  127770.0  97828.56    17
## 57   1.3944216 113012.5  14830.97  127843.5  98181.55    17
## 58   1.2705450 113222.7  14703.54  127926.2  98519.13    17
## 59   1.1576733 113454.2  14620.39  128074.6  98833.86    17
## 60   1.0548288 113675.1  14544.86  128220.0  99130.24    17
## 61   0.9611207 113889.1  14479.07  128368.1  99409.99    17
## 62   0.8757374 114083.4  14425.81  128509.2  99657.62    17
## 63   0.7979393 114259.6  14382.31  128642.0  99877.33    17
## 64   0.7270526 114434.1  14345.87  128779.9 100088.18    17
## 65   0.6624632 114591.4  14304.36  128895.7 100287.03    18
## 66   0.6036118 114737.1  14264.73  129001.9 100472.42    18
## 67   0.5499886 114867.9  14228.41  129096.3 100639.45    18
## 68   0.5011291 114985.0  14188.19  129173.2 100796.80    17
## 69   0.4566102 115067.7  14147.56  129215.2 100920.09    18
## 70   0.4160462 115144.5  14116.50  129261.0 101027.97    18
## 71   0.3790858 115227.7  14086.25  129314.0 101141.47    18
## 72   0.3454089 115296.0  14064.91  129360.9 101231.13    18
## 73   0.3147237 115367.9  14041.32  129409.2 101326.58    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.935124   69.40069
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
