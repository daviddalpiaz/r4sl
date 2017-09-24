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
## (Intercept) 172.720338908
## AtBat         0.099662970
## Hits          0.427613303
## HmRun         1.267838796
## Runs          0.676642660
## RBI           0.664847506
## Walks         0.887265880
## Years         2.665510665
## CAtBat        0.008472029
## CHits         0.033099124
## CHmRun        0.244686353
## CRuns         0.066354566
## CRBI          0.068696462
## CWalks        0.064445823
## LeagueN       4.803143606
## DivisionW   -27.147059583
## PutOuts       0.063770572
## Assists       0.008745578
## Errors       -0.209468235
## NewLeagueN    4.058198336
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
## (Intercept) 172.720338908
## AtBat         0.099662970
## Hits          0.427613303
## HmRun         1.267838796
## Runs          0.676642660
## RBI           0.664847506
## Walks         0.887265880
## Years         2.665510665
## CAtBat        0.008472029
## CHits         0.033099124
## CHmRun        0.244686353
## CRuns         0.066354566
## CRBI          0.068696462
## CWalks        0.064445823
## LeagueN       4.803143606
## DivisionW   -27.147059583
## PutOuts       0.063770572
## Assists       0.008745578
## Errors       -0.209468235
## NewLeagueN    4.058198336
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 787.2166
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 126796
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 450.9510 449.1953 448.4939 448.2248 447.9307 447.6093 447.2583
##  [8] 446.8751 446.4569 446.0008 445.5036 444.9622 444.3728 443.7317
## [15] 443.0351 442.2786 441.4582 440.5694 439.6076 438.5682 437.4465
## [22] 436.2378 434.9375 433.5412 432.0448 430.4444 428.7366 426.9188
## [29] 424.9888 422.9455 420.7888 418.5197 416.1406 413.6552 411.0688
## [36] 408.3884 405.6227 402.7816 399.8771 396.9225 393.9326 390.9232
## [43] 387.9108 384.9125 381.9456 379.0270 376.1730 373.3989 370.7184
## [50] 368.1436 365.6847 363.3495 361.1445 359.0735 357.1382 355.3383
## [57] 353.6719 352.1355 350.7244 349.4328 348.2543 347.1819 346.2085
## [64] 345.3245 344.5283 343.8130 343.1683 342.5878 342.0632 341.5949
## [71] 341.1782 340.8035 340.4654 340.1650 339.8938 339.6504 339.4295
## [78] 339.2280 339.0447 338.8746 338.7152 338.5650 338.4214 338.2835
## [85] 338.1476 338.0155 337.8818 337.7488 337.6159 337.4812 337.3448
## [92] 337.2070 337.0686 336.9315 336.7917 336.6556 336.5188 336.3889
## [99] 336.2598
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.2598
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 361.1445
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
##  [1] 450.4540 442.6775 433.7607 425.3735 417.8451 409.0247 400.7598
##  [8] 393.4372 387.2004 381.9729 377.3634 373.1130 369.4002 366.2756
## [15] 363.3898 360.8197 358.4513 356.1423 353.9650 352.0087 350.3455
## [22] 348.9492 347.7745 346.7871 345.9917 345.4072 344.9889 344.6939
## [29] 344.5342 344.4483 344.3854 344.3537 344.3379 344.3552 344.3823
## [36] 344.4628 344.9752 345.6247 345.9150 345.8998 345.4400 344.6346
## [43] 343.9149 343.1600 342.5741 342.1342 341.7686 341.5723 341.5427
## [50] 341.6468 341.8787 342.3381 342.8576 343.3622 343.8714 344.3881
## [57] 344.8128 345.1908 345.5716 346.0105 346.3603 346.7019 346.9437
## [64] 347.1506 347.3305 347.5023 347.6708 347.8428 347.9965 348.1427
## [71] 348.3157 348.4322 348.5553 348.6515
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.5427
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.4002
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
## 1  255.2820965 202908.8  35342.80  238251.6 167565.96     0
## 2  232.6035386 195963.3  35603.60  231566.9 160359.73     1
## 3  211.9396813 188148.3  34923.59  223071.9 153224.75     2
## 4  193.1115442 180942.6  33899.05  214841.7 147043.59     2
## 5  175.9560468 174594.5  33060.03  207654.6 141534.49     3
## 6  160.3245966 167301.2  32428.49  199729.7 134872.71     4
## 7  146.0818013 160608.4  31752.90  192361.3 128855.51     4
## 8  133.1042967 154792.8  31071.54  185864.4 123721.30     4
## 9  121.2796778 149924.2  30452.40  180376.6 119471.79     4
## 10 110.5055255 145903.3  29896.03  175799.3 116007.27     4
## 11 100.6885192 142403.2  29336.21  171739.4 113066.95     5
## 12  91.7436287 139213.3  28835.86  168049.2 110377.47     5
## 13  83.5933775 136456.5  28434.63  164891.1 108021.86     5
## 14  76.1671723 134157.8  28107.20  162265.0 106050.58     5
## 15  69.4006906 132052.1  27875.04  159927.2 104177.07     6
## 16  63.2353245 130190.8  27666.56  157857.4 102524.28     6
## 17  57.6176726 128487.4  27433.38  155920.8 101053.99     6
## 18  52.4990774 126837.3  27141.97  153979.3  99695.38     6
## 19  47.8352040 125291.2  26784.35  152075.6  98506.89     6
## 20  43.5856563 123910.1  26409.01  150319.1  97501.10     6
## 21  39.7136268 122742.0  26074.88  148816.8  96667.07     6
## 22  36.1855776 121765.6  25785.15  147550.7  95980.40     6
## 23  32.9709506 120947.1  25533.12  146480.2  95414.00     6
## 24  30.0419022 120261.3  25313.42  145574.7  94947.88     6
## 25  27.3730624 119710.2  25122.87  144833.1  94587.37     6
## 26  24.9413150 119306.1  24967.18  144273.3  94338.97     6
## 27  22.7255973 119017.4  24836.07  143853.4  94181.31     6
## 28  20.7067179 118813.9  24733.82  143547.7  94080.04     6
## 29  18.8671902 118703.8  24676.44  143380.3  94027.40     6
## 30  17.1910810 118644.6  24640.54  143285.1  94004.07     7
## 31  15.6638727 118601.3  24612.50  143213.8  93988.78     7
## 32  14.2723374 118579.5  24588.43  143167.9  93991.03     7
## 33  13.0044223 118568.6  24566.70  143135.3  94001.86     9
## 34  11.8491453 118580.5  24547.58  143128.1  94032.95     9
## 35  10.7964999 118599.2  24534.44  143133.6  94064.73     9
## 36   9.8373686 118654.6  24511.83  143166.4  94142.79     9
## 37   8.9634439 119007.9  24448.17  143456.0  94559.70     9
## 38   8.1671562 119456.4  24388.17  143844.6  95068.24    11
## 39   7.4416086 119657.2  24239.86  143897.1  95417.36    11
## 40   6.7805166 119646.7  24028.60  143675.3  95618.09    12
## 41   6.1781542 119328.8  23756.78  143085.6  95572.00    12
## 42   5.6293040 118773.0  23475.30  142248.3  95297.71    13
## 43   5.1292121 118277.5  23226.30  141503.8  95051.19    13
## 44   4.6735471 117758.8  23011.44  140770.2  94747.33    13
## 45   4.2583620 117357.0  22843.78  140200.8  94513.26    13
## 46   3.8800609 117055.8  22712.26  139768.1  94343.55    13
## 47   3.5353670 116805.7  22586.15  139391.9  94219.60    13
## 48   3.2212947 116671.7  22470.29  139142.0  94201.38    13
## 49   2.9351238 116651.4  22373.91  139025.3  94277.54    13
## 50   2.6743755 116722.5  22293.13  139015.7  94429.41    13
## 51   2.4367913 116881.0  22239.38  139120.4  94641.66    13
## 52   2.2203135 117195.4  22211.17  139406.6  94984.23    14
## 53   2.0230670 117551.3  22181.59  139732.9  95369.72    15
## 54   1.8433433 117897.6  22150.51  140048.1  95747.10    15
## 55   1.6795857 118247.6  22142.40  140390.0  96105.18    17
## 56   1.5303760 118603.2  22176.24  140779.4  96426.92    17
## 57   1.3944216 118895.9  22203.90  141099.8  96692.00    17
## 58   1.2705450 119156.7  22214.13  141370.8  96942.58    17
## 59   1.1576733 119419.8  22239.76  141659.5  97180.00    17
## 60   1.0548288 119723.3  22303.79  142027.1  97419.49    17
## 61   0.9611207 119965.5  22380.29  142345.8  97585.20    17
## 62   0.8757374 120202.2  22453.39  142655.6  97748.82    17
## 63   0.7979393 120370.0  22523.27  142893.2  97846.68    17
## 64   0.7270526 120513.5  22586.18  143099.7  97927.35    17
## 65   0.6624632 120638.5  22642.79  143281.3  97995.70    18
## 66   0.6036118 120757.8  22693.50  143451.3  98064.34    18
## 67   0.5499886 120875.0  22740.08  143615.0  98134.89    18
## 68   0.5011291 120994.6  22782.57  143777.2  98212.07    17
## 69   0.4566102 121101.6  22819.97  143921.5  98281.59    18
## 70   0.4160462 121203.3  22858.89  144062.2  98344.43    18
## 71   0.3790858 121323.8  22885.58  144209.4  98438.27    18
## 72   0.3454089 121405.0  22918.90  144323.9  98486.07    18
## 73   0.3147237 121490.8  22945.29  144436.1  98545.53    18
## 74   0.2867645 121557.9  22974.15  144532.0  98583.75    18
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
