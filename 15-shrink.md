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
##  [1] 452.4203 450.7792 450.0453 449.7731 449.4757 449.1506 448.7955
##  [8] 448.4079 447.9849 447.5236 447.0208 446.4731 445.8770 445.2287
## [15] 444.5242 443.7592 442.9295 442.0308 441.0582 440.0073 438.8731
## [22] 437.6510 436.3365 434.9250 433.4124 431.7948 430.0689 428.2320
## [29] 426.2819 424.2178 422.0394 419.7479 417.3457 414.8368 412.2268
## [36] 409.5227 406.7336 403.8695 400.9429 397.9675 394.9583 391.9316
## [43] 388.9043 385.8940 382.9183 379.9947 377.1397 374.3692 371.6973
## [50] 369.1364 366.6959 364.3849 362.2100 360.1747 358.2807 356.5278
## [57] 354.9138 353.4351 352.0869 350.8632 349.7572 348.7619 347.8689
## [64] 347.0713 346.3655 345.7406 345.1919 344.7094 344.2866 343.9236
## [71] 343.6117 343.3418 343.1125 342.9284 342.7684 342.6352 342.5304
## [78] 342.4447 342.3752 342.3183 342.2731 342.2342 342.2001 342.1683
## [85] 342.1371 342.1032 342.0650 342.0217 341.9706 341.9141 341.8475
## [92] 341.7721 341.6879 341.5951 341.4920 341.3812 341.2625 341.1368
## [99] 341.0052
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.0052
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.1364
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
## (Intercept)   10.7396773
## AtBat          .        
## Hits           1.9034076
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          2.2452717
## Years          .        
## CAtBat         .        
## CHits          .        
## CHmRun         .        
## CRuns          0.2097393
## CRBI           0.4164539
## CWalks         .        
## LeagueN        7.3925031
## DivisionW   -107.9671619
## PutOuts        0.2263070
## Assists        .        
## Errors         .        
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 120.3608
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
##  [1] 451.2131 441.7400 431.2809 422.5467 414.6254 405.9946 397.4185
##  [8] 389.9575 383.7523 378.5789 374.0817 370.0917 366.4510 363.0989
## [15] 359.9305 356.7904 353.8383 351.3168 349.2174 347.4717 346.0290
## [22] 344.8361 343.8502 343.0375 342.3972 341.9491 341.6315 341.4451
## [29] 341.3605 341.3257 341.3153 341.3025 341.3664 341.5754 341.9145
## [36] 342.4633 343.3419 343.9969 344.3526 344.2586 343.9949 343.6574
## [43] 343.4697 343.3236 343.1775 342.9002 342.5304 342.2619 342.1349
## [50] 342.0937 342.1305 342.2812 342.5794 342.9997 343.3694 343.6714
## [57] 343.9530 344.3274 344.7004 345.0652 345.4236 345.7776 346.1086
## [64] 346.4571 346.7657 347.0645 347.3427 347.6319 347.8866 348.1030
## [71] 348.2973 348.4916 348.6248 348.6887
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.3025
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 363.0989
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
## 1  255.2820965 203593.2  27619.05  231212.3 175974.19     0
## 2  232.6035386 195134.2  27051.12  222185.3 168083.10     1
## 3  211.9396813 186003.2  26070.00  212073.2 159933.21     2
## 4  193.1115442 178545.7  25255.59  203801.3 153290.08     2
## 5  175.9560468 171914.2  24644.29  196558.5 147269.96     3
## 6  160.3245966 164831.7  23934.07  188765.7 140897.58     4
## 7  146.0818013 157941.5  23047.66  180989.2 134893.84     4
## 8  133.1042967 152066.9  22253.28  174320.2 129813.59     4
## 9  121.2796778 147265.8  21595.87  168861.7 125669.94     4
## 10 110.5055255 143322.0  21009.98  164331.9 122311.99     4
## 11 100.6885192 139937.1  20533.71  160470.8 119403.42     5
## 12  91.7436287 136967.8  20127.60  157095.4 116840.23     5
## 13  83.5933775 134286.4  19734.75  154021.1 114551.61     5
## 14  76.1671723 131840.8  19346.00  151186.8 112494.79     5
## 15  69.4006906 129550.0  18975.72  148525.7 110574.24     6
## 16  63.2353245 127299.4  18639.56  145938.9 108659.80     6
## 17  57.6176726 125201.6  18337.89  143539.5 106863.67     6
## 18  52.4990774 123423.5  18084.72  141508.2 105338.74     6
## 19  47.8352040 121952.8  17875.88  139828.6 104076.88     6
## 20  43.5856563 120736.6  17704.06  138440.6 103032.51     6
## 21  39.7136268 119736.0  17565.01  137301.1 102171.03     6
## 22  36.1855776 118912.0  17452.53  136364.5 101459.43     6
## 23  32.9709506 118233.0  17360.95  135593.9 100872.02     6
## 24  30.0419022 117674.8  17286.96  134961.7 100387.80     6
## 25  27.3730624 117235.9  17231.44  134467.3 100004.44     6
## 26  24.9413150 116929.2  17198.17  134127.3  99731.00     6
## 27  22.7255973 116712.1  17175.11  133887.2  99536.97     6
## 28  20.7067179 116584.8  17146.80  133731.6  99437.99     6
## 29  18.8671902 116527.0  17117.29  133644.3  99409.70     6
## 30  17.1910810 116503.2  17096.87  133600.1  99406.33     7
## 31  15.6638727 116496.1  17082.80  133578.9  99413.31     7
## 32  14.2723374 116487.4  17073.41  133560.8  99413.97     7
## 33  13.0044223 116531.0  17086.13  133617.1  99444.87     9
## 34  11.8491453 116673.7  17105.18  133778.9  99568.54     9
## 35  10.7964999 116905.5  17108.69  134014.2  99796.83     9
## 36   9.8373686 117281.1  17053.36  134334.5 100227.75     9
## 37   8.9634439 117883.7  16932.78  134816.5 100950.91     9
## 38   8.1671562 118333.9  16793.87  135127.8 101540.03    11
## 39   7.4416086 118578.7  16656.53  135235.3 101922.20    11
## 40   6.7805166 118514.0  16524.82  135038.8 101989.13    12
## 41   6.1781542 118332.5  16411.52  134744.0 101920.96    12
## 42   5.6293040 118100.4  16301.69  134402.1 101798.73    13
## 43   5.1292121 117971.4  16232.99  134204.4 101738.45    13
## 44   4.6735471 117871.1  16177.49  134048.6 101693.61    13
## 45   4.2583620 117770.8  16130.55  133901.4 101640.25    13
## 46   3.8800609 117580.5  16059.00  133639.5 101521.54    13
## 47   3.5353670 117327.0  15974.83  133301.9 101352.22    13
## 48   3.2212947 117143.2  15893.37  133036.6 101249.84    13
## 49   2.9351238 117056.3  15802.41  132858.7 101253.88    13
## 50   2.6743755 117028.1  15696.35  132724.5 101331.77    13
## 51   2.4367913 117053.3  15583.15  132636.4 101470.13    13
## 52   2.2203135 117156.4  15477.39  132633.8 101679.00    14
## 53   2.0230670 117360.6  15371.65  132732.3 101988.98    15
## 54   1.8433433 117648.8  15274.57  132923.4 102374.24    15
## 55   1.6795857 117902.5  15185.43  133088.0 102717.09    17
## 56   1.5303760 118110.0  15084.38  133194.4 103025.65    17
## 57   1.3944216 118303.7  15004.48  133308.1 103299.17    17
## 58   1.2705450 118561.3  14938.26  133499.6 103623.07    17
## 59   1.1576733 118818.4  14875.58  133694.0 103942.81    17
## 60   1.0548288 119070.0  14814.57  133884.6 104255.42    17
## 61   0.9611207 119317.5  14760.88  134078.3 104556.58    17
## 62   0.8757374 119562.1  14712.01  134274.1 104850.12    17
## 63   0.7979393 119791.1  14662.76  134453.9 105128.39    17
## 64   0.7270526 120032.5  14612.62  134645.1 105419.90    17
## 65   0.6624632 120246.5  14566.95  134813.4 105679.52    18
## 66   0.6036118 120453.7  14528.06  134981.8 105925.69    18
## 67   0.5499886 120647.0  14489.85  135136.8 106157.11    18
## 68   0.5011291 120847.9  14456.87  135304.8 106391.04    17
## 69   0.4566102 121025.1  14434.36  135459.5 106590.76    18
## 70   0.4160462 121175.7  14404.24  135580.0 106771.47    18
## 71   0.3790858 121311.0  14382.85  135693.8 106928.13    18
## 72   0.3454089 121446.4  14361.97  135808.4 107084.45    18
## 73   0.3147237 121539.3  14348.77  135888.0 107190.50    18
## 74   0.2867645 121583.8  14339.44  135923.2 107244.36    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   14.27234   76.16717
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
