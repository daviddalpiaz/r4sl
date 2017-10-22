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
##  [1] 450.8483 449.1096 448.3320 448.0640 447.7710 447.4508 447.1011
##  [8] 446.7194 446.3028 445.8485 445.3533 444.8139 444.2269 443.5884
## [15] 442.8946 442.1412 441.3243 440.4392 439.4816 438.4467 437.3299
## [22] 436.1267 434.8324 433.4427 431.9534 430.3609 428.6617 426.8533
## [29] 424.9336 422.9016 420.7571 418.5013 416.1366 413.6668 411.0973
## [36] 408.4351 405.6890 402.8688 399.9866 397.0557 394.0908 391.1077
## [43] 388.1227 385.1530 382.2156 379.3274 376.5043 373.7614 371.1125
## [50] 368.5691 366.1411 363.8365 361.6611 359.6188 357.7109 355.9372
## [57] 354.2956 352.7826 351.3934 350.1223 348.9630 347.9086 346.9521
## [64] 346.0842 345.3045 344.6037 343.9744 343.4085 342.8995 342.4481
## [71] 342.0493 341.6952 341.3780 341.0996 340.8539 340.6370 340.4478
## [78] 340.2791 340.1307 340.0009 339.8856 339.7823 339.6897 339.6054
## [85] 339.5286 339.4569 339.3888 339.3245 339.2632 339.2029 339.1453
## [92] 339.0885 339.0332 338.9796 338.9277 338.8788 338.8331 338.7913
## [99] 338.7560
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.756
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 373.7614
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
## (Intercept)  156.0518715
## AtBat         -1.9655530
## Hits           6.9795719
## HmRun          1.6136362
## Runs          -1.3628757
## RBI            .        
## Walks          5.7647888
## Years         -6.8442567
## CAtBat        -0.0852048
## CHits          .        
## CHmRun         0.1105808
## CRuns          1.2363365
## CRBI           0.6280939
## CWalks        -0.7577815
## LeagueN       50.6138215
## DivisionW   -116.0638587
## PutOuts        0.2825782
## Assists        0.3082481
## Errors        -2.9916608
## NewLeagueN   -13.6905486
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 211.2994
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
##  [1] 453.3275 446.2872 438.3248 430.1461 422.7964 414.9154 406.4821
##  [8] 398.7997 392.2582 386.7819 382.2175 378.1862 374.6595 371.6214
## [15] 368.4320 364.7948 361.4947 358.6949 356.3527 354.3798 352.7177
## [22] 351.3179 350.1373 349.1434 348.3186 347.6261 347.0481 346.6430
## [29] 346.3530 346.1278 345.9806 345.8858 345.8371 345.8897 345.9839
## [36] 346.2457 346.7238 347.2347 347.3014 347.0259 346.6061 345.9322
## [43] 345.4092 344.9498 344.5361 344.2671 344.0049 343.5243 343.0948
## [50] 342.7569 342.5667 342.5009 342.5274 342.5703 342.5735 342.5738
## [57] 342.5394 342.5204 342.5031 342.4088 342.3167 342.3072 342.2923
## [64] 342.2821 342.2944 342.3436 342.3704 342.4184 342.4857 342.5172
## [71] 342.5972 342.6230 342.7032
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.2821
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 364.7948
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
## 1  255.2820965 205505.8  42926.55  248432.4 162579.29     0
## 2  232.6035386 199172.3  43634.95  242807.2 155537.32     1
## 3  211.9396813 192128.7  43042.26  235170.9 149086.41     2
## 4  193.1115442 185025.7  41122.17  226147.9 143903.53     2
## 5  175.9560468 178756.8  39440.35  218197.1 139316.41     3
## 6  160.3245966 172154.8  37880.06  210034.8 134274.69     4
## 7  146.0818013 165227.7  36131.70  201359.4 129095.96     4
## 8  133.1042967 159041.2  34446.69  193487.9 124594.55     4
## 9  121.2796778 153866.5  33001.18  186867.7 120865.34     4
## 10 110.5055255 149600.3  31790.32  181390.6 117809.94     4
## 11 100.6885192 146090.2  30704.24  176794.4 115385.94     5
## 12  91.7436287 143024.8  29672.57  172697.4 113352.24     5
## 13  83.5933775 140369.7  28741.03  169110.7 111628.68     5
## 14  76.1671723 138102.5  27905.14  166007.6 110197.33     5
## 15  69.4006906 135742.1  27104.42  162846.5 108637.71     6
## 16  63.2353245 133075.2  26220.72  159295.9 106854.51     6
## 17  57.6176726 130678.4  25448.88  156127.3 105229.57     6
## 18  52.4990774 128662.0  24787.55  153449.6 103874.45     6
## 19  47.8352040 126987.3  24224.38  151211.6 102762.89     6
## 20  43.5856563 125585.1  23734.30  149319.4 101850.75     6
## 21  39.7136268 124409.8  23309.01  147718.8 101100.78     6
## 22  36.1855776 123424.3  22938.57  146362.8 100485.71     6
## 23  32.9709506 122596.1  22616.76  145212.9  99979.36     6
## 24  30.0419022 121901.1  22335.00  144236.1  99566.13     6
## 25  27.3730624 121325.8  22086.50  143412.3  99239.32     6
## 26  24.9413150 120843.9  21868.39  142712.3  98975.55     6
## 27  22.7255973 120442.4  21675.86  142118.2  98766.49     6
## 28  20.7067179 120161.4  21497.55  141658.9  98663.82     6
## 29  18.8671902 119960.4  21342.57  141303.0  98617.80     6
## 30  17.1910810 119804.5  21210.17  141014.6  98594.30     7
## 31  15.6638727 119702.6  21101.70  140804.3  98600.89     7
## 32  14.2723374 119637.0  21015.51  140652.5  98621.49     7
## 33  13.0044223 119603.3  20949.81  140553.1  98653.45     9
## 34  11.8491453 119639.7  20919.18  140558.9  98720.53     9
## 35  10.7964999 119704.9  20896.53  140601.4  98808.34     9
## 36   9.8373686 119886.1  20851.46  140737.5  99034.60     9
## 37   8.9634439 120217.4  20790.24  141007.6  99427.15     9
## 38   8.1671562 120571.9  20682.60  141254.5  99889.35    11
## 39   7.4416086 120618.3  20517.73  141136.0 100100.53    11
## 40   6.7805166 120427.0  20326.60  140753.6 100100.41    12
## 41   6.1781542 120135.8  20156.04  140291.8  99979.72    12
## 42   5.6293040 119669.1  19971.54  139640.6  99697.56    13
## 43   5.1292121 119307.5  19815.05  139122.6  99492.48    13
## 44   4.6735471 118990.4  19671.92  138662.3  99318.43    13
## 45   4.2583620 118705.1  19555.70  138260.8  99149.45    13
## 46   3.8800609 118519.8  19463.21  137983.0  99056.60    13
## 47   3.5353670 118339.4  19353.84  137693.2  98985.52    13
## 48   3.2212947 118009.0  19161.93  137170.9  98847.03    13
## 49   2.9351238 117714.1  18979.30  136693.4  98734.76    13
## 50   2.6743755 117482.3  18821.73  136304.0  98660.54    13
## 51   2.4367913 117351.9  18685.10  136037.0  98666.84    13
## 52   2.2203135 117306.8  18565.66  135872.5  98741.18    14
## 53   2.0230670 117325.0  18457.18  135782.2  98867.81    15
## 54   1.8433433 117354.4  18359.26  135713.7  98995.16    15
## 55   1.6795857 117356.6  18284.82  135641.4  99071.78    17
## 56   1.5303760 117356.8  18201.51  135558.3  99155.28    17
## 57   1.3944216 117333.2  18105.10  135438.3  99228.12    17
## 58   1.2705450 117320.2  18019.75  135339.9  99300.44    17
## 59   1.1576733 117308.4  17932.03  135240.4  99376.36    17
## 60   1.0548288 117243.8  17817.57  135061.3  99426.20    17
## 61   0.9611207 117180.7  17699.72  134880.5  99481.01    17
## 62   0.8757374 117174.3  17600.59  134774.8  99573.66    17
## 63   0.7979393 117164.0  17514.57  134678.6  99649.43    17
## 64   0.7270526 117157.0  17429.58  134586.6  99727.46    17
## 65   0.6624632 117165.5  17356.51  134522.0  99808.96    18
## 66   0.6036118 117199.1  17294.11  134493.2  99905.01    18
## 67   0.5499886 117217.5  17237.00  134454.5  99980.46    18
## 68   0.5011291 117250.3  17188.33  134438.7 100062.01    17
## 69   0.4566102 117296.4  17145.51  134442.0 100150.93    18
## 70   0.4160462 117318.0  17097.23  134415.3 100220.81    18
## 71   0.3790858 117372.9  17063.80  134436.6 100309.05    18
## 72   0.3454089 117390.5  17027.28  134417.8 100363.21    18
## 73   0.3147237 117445.5  17006.51  134452.0 100438.94    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1  0.7270526   63.23532
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
