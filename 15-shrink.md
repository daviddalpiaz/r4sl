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

![](15-shrink_files/figure-latex/ridge-1.pdf)<!-- --> 

```r
plot(fit_ridge, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/ridge-2.pdf)<!-- --> 

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
##  [1] 451.6168 450.0856 449.3094 448.9842 448.6983 448.3858 448.0446
##  [8] 447.6720 447.2655 446.8222 446.3391 445.8128 445.2401 444.6172
## [15] 443.9404 443.2055 442.4087 441.5455 440.6116 439.6024 438.5136
## [22] 437.3405 436.0788 434.7243 433.2730 431.7213 430.0660 428.3045
## [29] 426.4351 424.4566 422.3693 420.1741 417.8735 415.4715 412.9733
## [36] 410.3858 407.7177 404.9790 402.1812 399.3376 396.4625 393.5713
## [43] 390.6803 387.8060 384.9651 382.1740 379.4483 376.8027 374.2504
## [50] 371.8029 369.4690 367.2575 365.1734 363.2194 361.3976 359.7074
## [57] 358.1464 356.7110 355.3962 354.1963 353.1047 352.1148 351.2193
## [64] 350.4108 349.6818 349.0293 348.4488 347.9245 347.4563 347.0402
## [71] 346.6724 346.3423 346.0502 345.7956 345.5653 345.3618 345.1821
## [78] 345.0206 344.8799 344.7447 344.6301 344.5131 344.4163 344.3126
## [85] 344.2242 344.1255 344.0363 343.9427 343.8527 343.7541 343.6594
## [92] 343.5575 343.4531 343.3448 343.2338 343.1201 343.0041 342.8859
## [99] 342.7639
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.7639
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 376.8027
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

![](15-shrink_files/figure-latex/lasso-1.pdf)<!-- --> 

```r
plot(fit_lasso, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/lasso-2.pdf)<!-- --> 

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

![](15-shrink_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

`cv.glmnet()` returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are again smaller than the full linear regression. (As we would expect.) Some coefficients are 0.


```r
coef(fit_lasso_cv)
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 193.74263858
## AtBat         .         
## Hits          1.21471320
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.28957902
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.12923755
## CRBI          0.31515925
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.02533305
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
## (Intercept) 193.74263858
## AtBat         .         
## Hits          1.21471320
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.28957902
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.12923755
## CRBI          0.31515925
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.02533305
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 2.974022
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 127112.4
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 451.2516 442.5431 432.5209 423.8596 415.6736 407.0688 398.5231
##  [8] 391.0085 384.6641 379.3728 375.1059 371.6113 368.6138 365.6702
## [15] 362.9104 360.3960 357.9200 355.6775 353.6124 351.7478 350.1156
## [22] 348.7519 347.6161 346.6810 345.9158 345.3288 344.9023 344.6261
## [29] 344.4266 344.2954 344.2221 344.2166 344.2555 344.3192 344.4192
## [36] 344.7742 345.4267 346.1867 346.7426 346.5526 346.2197 345.8237
## [43] 345.0845 344.3105 343.5530 342.9796 342.4804 342.1392 341.8888
## [50] 341.7429 341.6723 341.7511 341.8072 341.9284 342.0757 342.2803
## [57] 342.4298 342.6259 342.8451 343.0974 343.3159 343.5349 343.7888
## [64] 344.0095 344.2322 344.4825 344.7402 344.9775 345.2046 345.3348
## [71] 345.4402 345.5619
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.6723
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.6113
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
## 1  255.2820965 203628.0  30266.11  233894.1 173361.93     0
## 2  232.6035386 195844.4  29758.86  225603.3 166085.57     1
## 3  211.9396813 187074.3  28986.05  216060.4 158088.25     2
## 4  193.1115442 179656.9  28244.50  207901.4 151412.42     2
## 5  175.9560468 172784.5  27624.16  200408.7 145160.37     3
## 6  160.3245966 165705.0  27315.31  193020.3 138389.71     4
## 7  146.0818013 158820.6  27094.80  185915.4 131725.82     4
## 8  133.1042967 152887.7  26954.77  179842.4 125932.88     4
## 9  121.2796778 147966.5  26860.90  174827.4 121105.58     4
## 10 110.5055255 143923.7  26803.19  170726.9 117120.55     4
## 11 100.6885192 140704.5  26768.63  167473.1 113935.83     5
## 12  91.7436287 138095.0  26765.47  164860.5 111329.50     5
## 13  83.5933775 135876.1  26794.46  162670.6 109081.67     5
## 14  76.1671723 133714.7  26845.86  160560.5 106868.81     5
## 15  69.4006906 131704.0  26925.10  158629.1 104778.87     6
## 16  63.2353245 129885.2  27018.66  156903.9 102866.58     6
## 17  57.6176726 128106.7  26939.31  155046.0 101167.39     6
## 18  52.4990774 126506.5  26796.47  153303.0  99710.02     6
## 19  47.8352040 125041.7  26549.98  151591.7  98491.77     6
## 20  43.5856563 123726.5  26274.24  150000.8  97452.29     6
## 21  39.7136268 122580.9  26026.52  148607.4  96554.39     6
## 22  36.1855776 121627.9  25813.80  147441.7  95814.09     6
## 23  32.9709506 120836.9  25629.66  146466.6  95207.28     6
## 24  30.0419022 120187.7  25470.41  145658.1  94717.33     6
## 25  27.3730624 119657.7  25330.48  144988.2  94327.27     6
## 26  24.9413150 119252.0  25199.13  144451.1  94052.86     6
## 27  22.7255973 118957.6  25076.36  144034.0  93881.25     6
## 28  20.7067179 118767.1  24961.59  143728.7  93805.55     6
## 29  18.8671902 118629.7  24860.52  143490.2  93769.18     6
## 30  17.1910810 118539.3  24770.28  143309.6  93769.07     7
## 31  15.6638727 118488.9  24689.17  143178.1  93799.72     7
## 32  14.2723374 118485.0  24619.39  143104.4  93865.66     7
## 33  13.0044223 118511.8  24572.69  143084.5  93939.14     9
## 34  11.8491453 118555.7  24546.53  143102.3  94009.20     9
## 35  10.7964999 118624.6  24522.23  143146.8  94102.36     9
## 36   9.8373686 118869.3  24467.61  143336.9  94401.65     9
## 37   8.9634439 119319.6  24369.99  143689.6  94949.62     9
## 38   8.1671562 119845.3  24253.12  144098.4  95592.13    11
## 39   7.4416086 120230.4  24123.52  144354.0  96106.93    11
## 40   6.7805166 120098.7  23915.98  144014.7  96182.73    12
## 41   6.1781542 119868.1  23670.06  143538.1  96198.01    12
## 42   5.6293040 119594.1  23444.65  143038.7  96149.41    13
## 43   5.1292121 119083.3  23220.56  142303.9  95862.75    13
## 44   4.6735471 118549.7  23004.97  141554.7  95544.78    13
## 45   4.2583620 118028.6  22798.53  140827.2  95230.10    13
## 46   3.8800609 117635.0  22619.85  140254.9  95015.15    13
## 47   3.5353670 117292.8  22458.35  139751.2  94834.50    13
## 48   3.2212947 117059.3  22303.03  139362.3  94756.23    13
## 49   2.9351238 116887.9  22174.24  139062.2  94713.70    13
## 50   2.6743755 116788.2  22065.17  138853.4  94723.06    13
## 51   2.4367913 116740.0  21957.80  138697.7  94782.16    13
## 52   2.2203135 116793.8  21874.57  138668.4  94919.25    14
## 53   2.0230670 116832.1  21785.90  138618.0  95046.24    15
## 54   1.8433433 116915.0  21690.13  138605.1  95224.87    15
## 55   1.6795857 117015.8  21621.68  138637.5  95394.09    17
## 56   1.5303760 117155.8  21570.69  138726.5  95585.14    17
## 57   1.3944216 117258.1  21527.50  138785.6  95730.65    17
## 58   1.2705450 117392.5  21479.35  138871.9  95913.16    17
## 59   1.1576733 117542.7  21426.97  138969.7  96115.77    17
## 60   1.0548288 117715.8  21379.06  139094.9  96336.76    17
## 61   0.9611207 117865.8  21333.28  139199.1  96532.53    17
## 62   0.8757374 118016.2  21291.65  139307.9  96724.57    17
## 63   0.7979393 118190.7  21255.72  139446.5  96935.01    17
## 64   0.7270526 118342.5  21223.04  139565.6  97119.50    17
## 65   0.6624632 118495.8  21206.07  139701.9  97289.75    18
## 66   0.6036118 118668.2  21194.99  139863.2  97473.21    18
## 67   0.5499886 118845.8  21193.45  140039.2  97652.33    18
## 68   0.5011291 119009.5  21194.94  140204.4  97814.55    17
## 69   0.4566102 119166.2  21194.69  140360.9  97971.51    18
## 70   0.4160462 119256.1  21194.27  140450.4  98061.88    18
## 71   0.3790858 119329.0  21187.71  140516.7  98141.24    18
## 72   0.3454089 119413.0  21181.32  140594.3  98231.71    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   91.74363
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

![](15-shrink_files/figure-latex/unnamed-chunk-15-1.pdf)<!-- --> 


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

![](15-shrink_files/figure-latex/unnamed-chunk-19-1.pdf)<!-- --> 


```r
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```

![](15-shrink_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 

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

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.3.3 and the following packages:

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
