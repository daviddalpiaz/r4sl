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
## (Intercept) 213.066443434
## AtBat         0.090095728
## Hits          0.371252756
## HmRun         1.180126956
## Runs          0.596298287
## RBI           0.594502390
## Walks         0.772525466
## Years         2.473494238
## CAtBat        0.007597952
## CHits         0.029272172
## CHmRun        0.217335716
## CRuns         0.058705097
## CRBI          0.060722036
## CWalks        0.058698830
## LeagueN       3.276567828
## DivisionW   -21.889942619
## PutOuts       0.052667119
## Assists       0.007463678
## Errors       -0.145121336
## NewLeagueN    2.972759126
```

```r
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  6.713573e+01
## AtBat       -5.352890e-01
## Hits         2.397469e+00
## HmRun       -1.400213e+00
## Runs         1.095639e+00
## RBI          7.631426e-01
## Walks        3.083376e+00
## Years       -7.976859e+00
## CAtBat       2.442244e-03
## CHits        1.236025e-01
## CHmRun       6.715658e-01
## CRuns        2.592688e-01
## CRBI         2.400538e-01
## CWalks      -2.226824e-01
## LeagueN      5.043646e+01
## DivisionW   -1.213866e+02
## PutOuts      2.589908e-01
## Assists      1.490576e-01
## Errors      -3.546133e+00
## NewLeagueN  -1.472457e+01
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 17591.58
```

```r
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 213.066443434
## AtBat         0.090095728
## Hits          0.371252756
## HmRun         1.180126956
## Runs          0.596298287
## RBI           0.594502390
## Walks         0.772525466
## Years         2.473494238
## CAtBat        0.007597952
## CHits         0.029272172
## CHmRun        0.217335716
## CRuns         0.058705097
## CRBI          0.060722036
## CWalks        0.058698830
## LeagueN       3.276567828
## DivisionW   -21.889942619
## PutOuts       0.052667119
## Assists       0.007463678
## Errors       -0.145121336
## NewLeagueN    2.972759126
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 507.788
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 132355.6
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 450.9969 449.3900 448.5757 448.2967 448.0045 447.6853 447.3366
##  [8] 446.9560 446.5406 446.0876 445.5938 445.0560 444.4706 443.8340
## [15] 443.1421 442.3908 441.5760 440.6933 439.7382 438.7059 437.5920
## [22] 436.3917 435.1005 433.7141 432.2282 430.6391 428.9435 427.1387
## [29] 425.2226 423.1941 421.0532 418.8007 416.4392 413.9722 411.4052
## [36] 408.7451 406.0004 403.1811 400.2989 397.3673 394.4008 391.4150
## [43] 388.4265 385.4521 382.5089 379.6138 376.7830 374.0315 371.3730
## [50] 368.8198 366.3812 364.0663 361.8810 359.8292 357.9127 356.1317
## [57] 354.4843 352.9673 351.5765 350.3064 349.1508 348.1033 347.1571
## [64] 346.3050 345.5420 344.8639 344.2578 343.7206 343.2462 342.8312
## [71] 342.4705 342.1588 341.8881 341.6558 341.4637 341.2983 341.1595
## [78] 341.0491 340.9563 340.8838 340.8241 340.7787 340.7408 340.7138
## [85] 340.6912 340.6721 340.6585 340.6479 340.6368 340.6250 340.6155
## [92] 340.6064 340.5971 340.5892 340.5823 340.5766 340.5739 340.5742
## [99] 340.5775
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.5739
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 368.8198
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
## (Intercept) 144.37970458
## AtBat         .         
## Hits          1.36380384
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.49731098
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.15275165
## CRBI          0.32833941
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.06625755
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
## (Intercept)  123.7520756
## AtBat         -1.5473426
## Hits           5.6608972
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.7296908
## Years         -9.5958375
## CAtBat         .        
## CHits          .        
## CHmRun         0.5108207
## CRuns          0.6594856
## CRBI           0.3927505
## CWalks        -0.5291586
## LeagueN       32.0650811
## DivisionW   -119.2990171
## PutOuts        0.2724045
## Assists        0.1732025
## Errors        -2.0585083
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 177.4942
```

```r
coef(fit_lasso_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 144.37970458
## AtBat         .         
## Hits          1.36380384
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.49731098
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.15275165
## CRBI          0.32833941
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.06625755
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 3.408463
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 121290.9
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 449.5899 441.6352 432.8708 424.4095 416.8191 408.5193 400.6020
##  [8] 393.3075 386.7077 381.1476 376.5440 372.5909 369.1249 365.7467
## [15] 362.5711 359.6192 356.8285 354.3789 352.3168 350.6397 349.2589
## [22] 348.1396 347.2347 346.5189 345.9351 345.5182 345.2266 345.0731
## [29] 345.0267 345.0694 345.1108 345.1536 345.1853 345.3111 345.6492
## [36] 346.1244 346.7630 347.0096 346.5019 345.5692 344.6600 343.9128
## [43] 343.1238 342.2182 341.3284 340.5665 339.9332 339.5058 339.3822
## [50] 339.3142 339.3608 339.5835 339.9812 340.4463 340.8872 341.3214
## [57] 341.7190 341.9894 342.1891 342.4246 342.6611 342.8872 343.1177
## [64] 343.3349 343.5500 343.7667 343.9852 344.2211 344.4531 344.6629
## [71] 344.8200 345.0065 345.1625 345.3063 345.4769
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.3142
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 365.7467
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
## 1  255.2820965 202131.1  29638.60  231769.7 172492.47     0
## 2  232.6035386 195041.6  29781.55  224823.2 165260.09     1
## 3  211.9396813 187377.1  29116.89  216494.0 158260.26     2
## 4  193.1115442 180123.4  27937.95  208061.3 152185.43     2
## 5  175.9560468 173738.1  26935.97  200674.1 146802.16     3
## 6  160.3245966 166888.1  26137.18  193025.2 140750.87     4
## 7  146.0818013 160482.0  25506.65  185988.6 134975.31     4
## 8  133.1042967 154690.8  24953.70  179644.5 129737.06     4
## 9  121.2796778 149542.9  24458.03  174000.9 125084.84     4
## 10 110.5055255 145273.5  24037.92  169311.4 121235.59     4
## 11 100.6885192 141785.4  23694.14  165479.6 118091.29     5
## 12  91.7436287 138824.0  23428.00  162252.0 115396.01     5
## 13  83.5933775 136253.2  23221.06  159474.3 113032.16     5
## 14  76.1671723 133770.7  22988.59  156759.3 110782.07     5
## 15  69.4006906 131457.8  22797.41  154255.2 108660.41     6
## 16  63.2353245 129326.0  22661.61  151987.6 106664.34     6
## 17  57.6176726 127326.6  22508.53  149835.1 104818.07     6
## 18  52.4990774 125584.4  22358.56  147943.0 103225.83     6
## 19  47.8352040 124127.1  22233.52  146360.7 101893.63     6
## 20  43.5856563 122948.2  22138.70  145086.9 100809.53     6
## 21  39.7136268 121981.8  22071.63  144053.4  99910.18     6
## 22  36.1855776 121201.1  22024.84  143226.0  99176.31     6
## 23  32.9709506 120571.9  21994.52  142566.5  98577.41     6
## 24  30.0419022 120075.4  21979.42  142054.8  98095.96     6
## 25  27.3730624 119671.1  21974.58  141645.7  97696.51     6
## 26  24.9413150 119382.8  21973.27  141356.1  97409.56     6
## 27  22.7255973 119181.4  21978.34  141159.7  97203.03     6
## 28  20.7067179 119075.4  21994.92  141070.3  97080.52     6
## 29  18.8671902 119043.4  22006.66  141050.1  97036.75     6
## 30  17.1910810 119072.9  22013.22  141086.1  97059.64     7
## 31  15.6638727 119101.4  22021.49  141122.9  97079.95     7
## 32  14.2723374 119131.0  22029.98  141161.0  97101.01     7
## 33  13.0044223 119152.9  22037.02  141189.9  97115.88     9
## 34  11.8491453 119239.8  22034.49  141274.3  97205.30     9
## 35  10.7964999 119473.4  22008.80  141482.2  97464.59     9
## 36   9.8373686 119802.1  21963.52  141765.7  97838.60     9
## 37   8.9634439 120244.6  21891.30  142135.9  98353.31     9
## 38   8.1671562 120415.7  21845.84  142261.5  98569.82    11
## 39   7.4416086 120063.5  21775.18  141838.7  98288.36    11
## 40   6.7805166 119418.1  21587.34  141005.4  97830.74    12
## 41   6.1781542 118790.5  21357.75  140148.2  97432.74    12
## 42   5.6293040 118276.0  21139.95  139415.9  97136.05    13
## 43   5.1292121 117733.9  20932.45  138666.4  96801.50    13
## 44   4.6735471 117113.3  20714.32  137827.7  96399.00    13
## 45   4.2583620 116505.1  20507.19  137012.3  95997.88    13
## 46   3.8800609 115985.6  20311.66  136297.2  95673.90    13
## 47   3.5353670 115554.6  20135.61  135690.2  95418.95    13
## 48   3.2212947 115264.2  19985.47  135249.6  95278.69    13
## 49   2.9351238 115180.3  19849.43  135029.7  95330.88    13
## 50   2.6743755 115134.1  19727.37  134861.5  95406.78    13
## 51   2.4367913 115165.7  19611.42  134777.1  95554.31    13
## 52   2.2203135 115317.0  19500.98  134818.0  95816.00    14
## 53   2.0230670 115587.2  19397.56  134984.8  96189.68    15
## 54   1.8433433 115903.7  19306.43  135210.1  96597.28    15
## 55   1.6795857 116204.1  19234.13  135438.2  96969.93    17
## 56   1.5303760 116500.3  19184.46  135684.7  97315.82    17
## 57   1.3944216 116771.9  19158.35  135930.2  97613.54    17
## 58   1.2705450 116956.7  19116.05  136072.8  97840.69    17
## 59   1.1576733 117093.4  19062.90  136156.3  98030.47    17
## 60   1.0548288 117254.6  19015.54  136270.1  98239.07    17
## 61   0.9611207 117416.6  18976.17  136392.8  98440.43    17
## 62   0.8757374 117571.7  18950.41  136522.1  98621.24    17
## 63   0.7979393 117729.7  18937.86  136667.6  98791.86    17
## 64   0.7270526 117878.9  18929.90  136808.8  98948.98    17
## 65   0.6624632 118026.6  18924.40  136951.0  99102.20    18
## 66   0.6036118 118175.6  18920.05  137095.6  99255.52    18
## 67   0.5499886 118325.8  18916.52  137242.3  99409.27    18
## 68   0.5011291 118488.1  18914.72  137402.9  99573.42    17
## 69   0.4566102 118647.9  18914.55  137562.5  99733.39    18
## 70   0.4160462 118792.5  18909.34  137701.9  99883.18    18
## 71   0.3790858 118900.8  18904.29  137805.1  99996.51    18
## 72   0.3454089 119029.5  18898.97  137928.5 100130.52    18
## 73   0.3147237 119137.1  18895.12  138032.3 100242.02    18
## 74   0.2867645 119236.5  18891.52  138128.0 100344.94    18
## 75   0.2612891 119354.3  18886.33  138240.6 100467.94    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   76.16717
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
