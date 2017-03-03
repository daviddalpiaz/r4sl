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
##  [1] 451.9539 449.9529 449.4158 449.1457 448.8506 448.5280 448.1757
##  [8] 447.7910 447.3712 446.9134 446.4143 445.8707 445.2789 444.6353
## [15] 443.9358 443.1761 442.3523 441.4596 440.4935 439.4494 438.3224
## [22] 437.1079 435.8012 434.3978 432.8935 431.2844 429.5670 427.7386
## [29] 425.7969 423.7407 421.5698 419.2851 416.8888 414.3845 411.7776
## [36] 409.0748 406.2847 403.4171 400.4840 397.4987 394.4758 391.4310
## [43] 388.3810 385.3429 382.3340 379.3714 376.4714 373.6495 370.9197
## [50] 368.2943 365.7830 363.3949 361.1362 359.0110 357.0212 355.1669
## [57] 353.4462 351.8560 350.3916 349.0474 347.8170 346.6935 345.6698
## [64] 344.7393 343.8950 343.1322 342.4392 341.8118 341.2456 340.7349
## [71] 340.2720 339.8511 339.4702 339.1260 338.8105 338.5207 338.2563
## [78] 338.0091 337.7796 337.5621 337.3583 337.1600 336.9712 336.7844
## [85] 336.6016 336.4230 336.2437 336.0651 335.8868 335.7077 335.5280
## [92] 335.3472 335.1670 334.9876 334.8106 334.6328 334.4608 334.2905
## [99] 334.1266
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 334.1266
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 359.011
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
##                       1
## (Intercept) 115.3773590
## AtBat         .        
## Hits          1.4753071
## HmRun         .        
## Runs          .        
## RBI           .        
## Walks         1.6566947
## Years         .        
## CAtBat        .        
## CHits         .        
## CHmRun        .        
## CRuns         0.1660465
## CRBI          0.3453397
## CWalks        .        
## LeagueN       .        
## DivisionW   -19.2435216
## PutOuts       0.1000068
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
##                       1
## (Intercept) 115.3773590
## AtBat         .        
## Hits          1.4753071
## HmRun         .        
## Runs          .        
## RBI           .        
## Walks         1.6566947
## Years         .        
## CAtBat        .        
## CHits         .        
## CHmRun        .        
## CRuns         0.1660465
## CRBI          0.3453397
## CWalks        .        
## LeagueN       .        
## DivisionW   -19.2435216
## PutOuts       0.1000068
## Assists       .        
## Errors        .        
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 22.98692
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 116096.9
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 449.1356 442.2315 431.8120 422.9605 415.0828 406.9291 398.6080
##  [8] 391.2634 384.7549 379.0127 374.0551 369.8045 365.9257 362.3576
## [15] 358.8731 355.5528 352.4403 349.7371 347.4726 345.5794 344.0049
## [22] 342.6958 341.6002 340.6910 339.9655 339.5626 339.2544 339.0328
## [29] 338.9217 338.8863 338.8592 338.8219 338.7676 338.7309 338.7723
## [36] 339.0288 339.3909 339.5781 339.6349 339.0072 337.7949 336.5201
## [43] 335.4164 334.4825 333.6524 332.9819 332.5421 332.2869 332.2232
## [50] 332.3181 332.4469 332.6045 332.7564 332.8605 333.0448 333.3179
## [57] 333.6811 334.1504 334.6569 335.1534 335.6281 336.0507 336.3853
## [64] 336.6341 336.8248 337.0083 337.1490 337.3278 337.5060 337.6713
## [71] 337.8299 337.9922 338.1415
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 332.2232
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 355.5528
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
## 1  255.2820965 201722.8  34930.95  236653.8 166791.87     0
## 2  232.6035386 195568.7  35407.61  230976.3 160161.08     1
## 3  211.9396813 186461.6  33778.45  220240.0 152683.11     2
## 4  193.1115442 178895.6  32384.48  211280.1 146511.12     2
## 5  175.9560468 172293.8  31231.24  203525.0 141062.51     3
## 6  160.3245966 165591.3  30120.13  195711.4 135471.15     4
## 7  146.0818013 158888.4  28883.76  187772.1 130004.61     4
## 8  133.1042967 153087.0  27907.23  180994.2 125179.78     4
## 9  121.2796778 148036.4  27085.93  175122.3 120950.42     4
## 10 110.5055255 143650.6  26254.78  169905.4 117395.82     4
## 11 100.6885192 139917.2  25485.84  165403.1 114431.40     5
## 12  91.7436287 136755.3  24854.85  161610.2 111900.48     5
## 13  83.5933775 133901.7  24312.68  158214.3 109588.97     5
## 14  76.1671723 131303.0  23826.93  155130.0 107476.10     5
## 15  69.4006906 128789.9  23332.01  152121.9 105457.89     6
## 16  63.2353245 126417.8  22875.32  149293.1 103542.46     6
## 17  57.6176726 124214.2  22413.42  146627.6 101800.77     6
## 18  52.4990774 122316.1  22010.30  144326.4 100305.77     6
## 19  47.8352040 120737.2  21670.33  142407.6  99066.90     6
## 20  43.5856563 119425.1  21382.58  140807.7  98042.53     6
## 21  39.7136268 118339.4  21136.92  139476.3  97202.47     6
## 22  36.1855776 117440.4  20926.69  138367.1  96513.71     6
## 23  32.9709506 116690.7  20746.84  137437.5  95943.85     6
## 24  30.0419022 116070.4  20591.16  136661.5  95479.23     6
## 25  27.3730624 115576.5  20455.18  136031.7  95121.35     6
## 26  24.9413150 115302.8  20318.83  135621.6  94983.93     6
## 27  22.7255973 115093.5  20194.55  135288.1  94898.98     6
## 28  20.7067179 114943.2  20081.34  135024.6  94861.88     6
## 29  18.8671902 114867.9  19975.48  134843.4  94892.42     6
## 30  17.1910810 114843.9  19879.26  134723.2  94964.64     7
## 31  15.6638727 114825.5  19792.49  134618.0  95033.06     7
## 32  14.2723374 114800.3  19712.92  134513.2  95087.36     7
## 33  13.0044223 114763.5  19632.72  134396.2  95130.77     9
## 34  11.8491453 114738.6  19560.88  134299.5  95177.77     9
## 35  10.7964999 114766.6  19493.67  134260.3  95272.97     9
## 36   9.8373686 114940.5  19431.77  134372.3  95508.77     9
## 37   8.9634439 115186.2  19388.11  134574.3  95798.09     9
## 38   8.1671562 115313.3  19358.37  134671.6  95954.89    11
## 39   7.4416086 115351.9  19323.45  134675.3  96028.40    11
## 40   6.7805166 114925.9  19253.71  134179.6  95672.16    12
## 41   6.1781542 114105.4  19023.18  133128.6  95082.23    12
## 42   5.6293040 113245.8  18700.95  131946.8  94544.86    13
## 43   5.1292121 112504.2  18395.57  130899.7  94108.59    13
## 44   4.6735471 111878.5  18120.33  129998.9  93758.21    13
## 45   4.2583620 111323.9  17877.10  129201.0  93446.81    13
## 46   3.8800609 110877.0  17654.01  128531.0  93222.96    13
## 47   3.5353670 110584.2  17448.18  128032.4  93136.05    13
## 48   3.2212947 110414.6  17263.91  127678.5  93150.66    13
## 49   2.9351238 110372.3  17089.70  127462.0  93282.55    13
## 50   2.6743755 110435.3  16934.39  127369.7  93500.95    13
## 51   2.4367913 110520.9  16774.06  127295.0  93746.89    13
## 52   2.2203135 110625.8  16602.80  127228.5  94022.95    14
## 53   2.0230670 110726.8  16463.34  127190.2  94263.48    15
## 54   1.8433433 110796.1  16330.34  127126.5  94465.79    15
## 55   1.6795857 110918.8  16221.88  127140.7  94696.96    17
## 56   1.5303760 111100.8  16130.10  127230.9  94970.75    17
## 57   1.3944216 111343.1  16046.41  127389.5  95296.64    17
## 58   1.2705450 111656.5  15967.41  127623.9  95689.10    17
## 59   1.1576733 111995.2  15898.75  127894.0  96096.46    17
## 60   1.0548288 112327.8  15837.52  128165.3  96490.25    17
## 61   0.9611207 112646.2  15785.26  128431.5  96860.95    17
## 62   0.8757374 112930.1  15741.64  128671.7  97188.41    17
## 63   0.7979393 113155.0  15707.42  128862.5  97447.62    17
## 64   0.7270526 113322.5  15688.08  129010.6  97634.42    17
## 65   0.6624632 113450.9  15677.34  129128.3  97773.58    18
## 66   0.6036118 113574.6  15674.51  129249.1  97900.11    18
## 67   0.5499886 113669.5  15677.58  129347.0  97991.88    18
## 68   0.5011291 113790.1  15685.58  129475.7  98104.50    17
## 69   0.4566102 113910.3  15691.03  129601.3  98219.27    18
## 70   0.4160462 114021.9  15699.97  129721.9  98321.94    18
## 71   0.3790858 114129.1  15708.98  129838.0  98420.09    18
## 72   0.3454089 114238.7  15717.71  129956.5  98521.04    18
## 73   0.3147237 114339.6  15723.55  130063.2  98616.09    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.935124   63.23532
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
- [`glmnet` with `caret`](https://github.com/topepo/caret/issues/116) - Some details on Elastic Net tuning in the `caret` package. TODO: move this to elastic net chapter.


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
