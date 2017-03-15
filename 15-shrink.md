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
##  [1] 451.5034 450.2417 449.2361 448.8849 448.5954 448.2791 447.9336
##  [8] 447.5564 447.1448 446.6959 446.2067 445.6738 445.0938 444.4631
## [15] 443.7776 443.0333 442.2261 441.3517 440.4056 439.3831 438.2798
## [22] 437.0909 435.8122 434.4392 432.9678 431.3943 429.7156 427.9288
## [29] 426.0321 424.0243 421.9054 419.6765 417.3399 414.8994 412.3603
## [36] 409.7295 407.0157 404.2287 401.3802 398.4835 395.5532 392.6048
## [43] 389.6547 386.7197 383.8169 380.9630 378.1741 375.4652 372.8499
## [50] 370.3405 367.9466 365.6767 363.5368 361.5301 359.6595 357.9248
## [57] 356.3241 354.8542 353.5106 352.2877 351.1793 350.1788 349.2790
## [64] 348.4711 347.7511 347.1144 346.5521 346.0540 345.6147 345.2314
## [71] 344.9026 344.6150 344.3689 344.1575 343.9758 343.8180 343.6907
## [78] 343.5771 343.4795 343.3957 343.3210 343.2529 343.1889 343.1257
## [85] 343.0646 343.0004 342.9323 342.8606 342.7838 342.7002 342.6107
## [92] 342.5148 342.4107 342.3016 342.1867 342.0677 341.9441 341.8177
## [99] 341.6894
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.6894
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 375.4652
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
##  [1] 449.3387 441.6263 433.1297 424.6154 417.0161 408.1963 399.5048
##  [8] 391.8593 385.3615 379.7969 375.1218 371.1893 367.6662 364.6715
## [15] 361.7325 358.8080 356.3107 354.1695 352.2018 350.5062 349.1266
## [22] 347.9933 347.0603 346.2946 345.6991 345.3009 345.0230 344.8309
## [29] 344.6995 344.6340 344.6366 344.6644 344.7820 345.0271 345.3870
## [36] 345.9567 346.6592 347.1407 347.0300 346.2970 345.4306 344.5415
## [43] 343.6933 343.0217 342.5348 342.1726 341.8957 341.6978 341.5864
## [50] 341.5796 341.6821 341.9358 342.2290 342.4936 342.8506 343.3001
## [57] 343.7511 344.1528 344.5439 344.9042 345.1805 345.4063 345.6573
## [64] 345.9131 346.1460 346.3079 346.4510 346.6155 346.7941 346.9733
## [71] 347.1186 347.2498 347.3726 347.5049 347.5856 347.6845
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.5796
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.6662
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
## 1  255.2820965 201905.2  25704.89  227610.1 176200.35     0
## 2  232.6035386 195033.8  25926.08  220959.8 169107.69     1
## 3  211.9396813 187601.3  25279.90  212881.2 162321.44     2
## 4  193.1115442 180298.3  23948.91  204247.2 156349.35     2
## 5  175.9560468 173902.4  22841.34  196743.7 151061.05     3
## 6  160.3245966 166624.2  21759.07  188383.3 144865.17     4
## 7  146.0818013 159604.1  20791.02  180395.1 138813.07     4
## 8  133.1042967 153553.7  19974.39  173528.1 133579.34     4
## 9  121.2796778 148503.5  19344.69  167848.2 129158.82     4
## 10 110.5055255 144245.7  18873.32  163119.0 125372.35     4
## 11 100.6885192 140716.3  18518.03  159234.4 122198.31     5
## 12  91.7436287 137781.5  18251.83  156033.4 119529.69     5
## 13  83.5933775 135178.4  18087.69  153266.1 117090.71     5
## 14  76.1671723 132985.3  17985.14  150970.4 115000.15     5
## 15  69.4006906 130850.4  17924.02  148774.4 112926.39     6
## 16  63.2353245 128743.2  17902.64  146645.8 110840.52     6
## 17  57.6176726 126957.3  17939.00  144896.3 109018.35     6
## 18  52.4990774 125436.0  18018.82  143454.9 107417.21     6
## 19  47.8352040 124046.1  18049.57  142095.7 105996.55     6
## 20  43.5856563 122854.6  18062.47  140917.0 104792.09     6
## 21  39.7136268 121889.4  18112.19  140001.6 103777.19     6
## 22  36.1855776 121099.3  18180.32  139279.7 102919.02     6
## 23  32.9709506 120450.9  18258.56  138709.4 102192.30     6
## 24  30.0419022 119920.0  18343.64  138263.6 101576.31     6
## 25  27.3730624 119507.9  18433.37  137941.3 101074.51     6
## 26  24.9413150 119232.7  18525.94  137758.7 100706.77     6
## 27  22.7255973 119040.9  18616.40  137657.3 100424.48     6
## 28  20.7067179 118908.4  18703.05  137611.4 100205.33     6
## 29  18.8671902 118817.7  18786.34  137604.1 100031.38     6
## 30  17.1910810 118772.6  18863.45  137636.1  99909.17     7
## 31  15.6638727 118774.4  18934.65  137709.0  99839.74     7
## 32  14.2723374 118793.5  19001.03  137794.6  99792.52     7
## 33  13.0044223 118874.6  19049.06  137923.7  99825.55     9
## 34  11.8491453 119043.7  19070.20  138113.9  99973.49     9
## 35  10.7964999 119292.2  19074.85  138367.0 100217.32     9
## 36   9.8373686 119686.0  19035.20  138721.2 100650.82     9
## 37   8.9634439 120172.6  18980.65  139153.3 101191.97     9
## 38   8.1671562 120506.6  18956.08  139462.7 101550.55    11
## 39   7.4416086 120429.8  18879.46  139309.3 101550.35    11
## 40   6.7805166 119921.6  18836.53  138758.2 101085.09    12
## 41   6.1781542 119322.3  18814.51  138136.8 100507.77    12
## 42   5.6293040 118708.8  18829.90  137538.7  99878.92    13
## 43   5.1292121 118125.1  18871.44  136996.5  99253.61    13
## 44   4.6735471 117663.9  18889.99  136553.9  98773.91    13
## 45   4.2583620 117330.1  18927.60  136257.7  98402.50    13
## 46   3.8800609 117082.1  18969.83  136051.9  98112.26    13
## 47   3.5353670 116892.7  18980.33  135873.0  97912.34    13
## 48   3.2212947 116757.4  18995.25  135752.7  97762.15    13
## 49   2.9351238 116681.3  19010.83  135692.1  97670.44    13
## 50   2.6743755 116676.6  19027.07  135703.7  97649.54    13
## 51   2.4367913 116746.7  19037.19  135783.9  97709.48    13
## 52   2.2203135 116920.1  19048.01  135968.1  97872.06    14
## 53   2.0230670 117120.7  19051.85  136172.5  98068.81    15
## 54   1.8433433 117301.9  19065.11  136367.0  98236.77    15
## 55   1.6795857 117546.5  19067.14  136613.7  98479.39    17
## 56   1.5303760 117855.0  19074.77  136929.8  98780.22    17
## 57   1.3944216 118164.8  19096.04  137260.9  99068.78    17
## 58   1.2705450 118441.2  19128.26  137569.4  99312.90    17
## 59   1.1576733 118710.5  19169.82  137880.3  99540.65    17
## 60   1.0548288 118958.9  19196.84  138155.8  99762.07    17
## 61   0.9611207 119149.5  19196.26  138345.8  99953.29    17
## 62   0.8757374 119305.5  19184.64  138490.2 100120.90    17
## 63   0.7979393 119479.0  19169.56  138648.5 100309.41    17
## 64   0.7270526 119655.8  19152.78  138808.6 100503.07    17
## 65   0.6624632 119817.0  19140.77  138957.8 100676.27    18
## 66   0.6036118 119929.1  19130.33  139059.5 100798.82    18
## 67   0.5499886 120028.3  19124.60  139152.9 100903.66    18
## 68   0.5011291 120142.3  19131.60  139273.9 101010.74    17
## 69   0.4566102 120266.1  19139.16  139405.3 101126.96    18
## 70   0.4160462 120390.5  19150.95  139541.4 101239.52    18
## 71   0.3790858 120491.4  19160.52  139651.9 101330.84    18
## 72   0.3454089 120582.4  19168.92  139751.3 101413.49    18
## 73   0.3147237 120667.7  19179.95  139847.7 101487.79    18
## 74   0.2867645 120759.6  19191.46  139951.1 101568.17    18
## 75   0.2612891 120815.8  19200.55  140016.3 101615.20    18
## 76   0.2380769 120884.5  19209.22  140093.7 101675.30    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   83.59338
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
