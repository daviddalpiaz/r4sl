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

![](14-shrink_files/figure-latex/ridge-1.pdf)<!-- --> 

```r
plot(fit_ridge, xvar = "lambda", label = TRUE)
```

![](14-shrink_files/figure-latex/ridge-2.pdf)<!-- --> 

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

![](14-shrink_files/figure-latex/unnamed-chunk-8-1.pdf)<!-- --> 

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
## (Intercept)   71.78758429
## AtBat         -0.58269657
## Hits           2.51715272
## HmRun         -1.39973428
## Runs           1.07259572
## RBI            0.74825248
## Walks          3.17950553
## Years         -8.35976899
## CAtBat         0.00133718
## CHits          0.12772556
## CHmRun         0.68074413
## CRuns          0.27080732
## CRBI           0.24581306
## CWalks        -0.24120197
## LeagueN       51.41107146
## DivisionW   -121.93563378
## PutOuts        0.26073685
## Assists        0.15595798
## Errors        -3.59749877
## NewLeagueN   -15.89754187
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 17868.18
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
##  [1] 450.4970 448.8719 448.0984 447.8312 447.5392 447.2201 446.8716
##  [8] 446.4911 446.0760 445.6233 445.1299 444.5925 444.0076 443.3716
## [15] 442.6804 441.9301 441.1164 440.2351 439.2815 438.2512 437.1395
## [22] 435.9419 434.6539 433.2712 431.7897 430.2059 428.5164 426.7186
## [29] 424.8108 422.7919 420.6620 418.4223 416.0753 413.6250 411.0768
## [36] 408.4380 405.7172 402.9247 400.0723 397.1736 394.2432 391.2969
## [43] 388.3511 385.4228 382.5290 379.6863 376.9107 374.2170 371.6191
## [50] 369.1278 366.7530 364.5025 362.3823 360.3956 358.5437 356.8262
## [57] 355.2408 353.7839 352.4505 351.2350 350.1308 349.1309 348.2283
## [64] 347.4132 346.6851 346.0372 345.4567 344.9393 344.4764 344.0728
## [71] 343.7177 343.3988 343.1248 342.8843 342.6699 342.4862 342.3222
## [78] 342.1831 342.0544 341.9447 341.8419 341.7508 341.6633 341.5795
## [85] 341.4975 341.4190 341.3369 341.2521 341.1650 341.0733 340.9786
## [92] 340.8783 340.7730 340.6631 340.5484 340.4295 340.3081 340.1836
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.1836
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 362.3823
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

![](14-shrink_files/figure-latex/lasso-1.pdf)<!-- --> 

```r
plot(fit_lasso, xvar = "lambda", label = TRUE)
```

![](14-shrink_files/figure-latex/lasso-2.pdf)<!-- --> 

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

![](14-shrink_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

`cv.glmnet()` returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are again smaller than the full linear regression. (As we would expect.) Some coefficients are 0.


```r
coef(fit_lasso_cv)
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                       1
## (Intercept)  93.4854094
## AtBat         .        
## Hits          1.5657087
## HmRun         .        
## Runs          .        
## RBI           .        
## Walks         1.7868806
## Years         .        
## CAtBat        .        
## CHits         .        
## CHmRun        .        
## CRuns         0.1755955
## CRBI          0.3608949
## CWalks        .        
## LeagueN       .        
## DivisionW   -38.7228685
## PutOuts       0.1279638
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
##                       1
## (Intercept)  93.4854094
## AtBat         .        
## Hits          1.5657087
## HmRun         .        
## Runs          .        
## RBI           .        
## Walks         1.7868806
## Years         .        
## CAtBat        .        
## CHits         .        
## CHmRun        .        
## CRuns         0.1755955
## CRBI          0.3608949
## CWalks        .        
## LeagueN       .        
## DivisionW   -38.7228685
## PutOuts       0.1279638
## Assists       .        
## Errors        .        
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 42.73991
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 112322.3
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 451.1128 441.1475 430.9873 422.6877 415.0665 407.0049 399.2427
##  [8] 392.2459 386.2632 381.1506 376.9441 373.3211 370.3101 367.3569
## [15] 364.5315 361.5350 358.8007 356.5235 354.5657 352.8125 351.3186
## [22] 350.0831 349.0568 348.1698 347.4172 346.7901 346.3346 346.0454
## [29] 345.8315 345.6863 345.5938 345.5434 345.5302 345.5045 345.5050
## [36] 345.5413 345.6286 345.6791 345.5151 345.1019 344.1394 342.9448
## [43] 341.7297 340.7118 339.8710 339.2019 338.6505 338.1541 337.8373
## [50] 337.6774 337.6756 337.8100 337.9588 338.1627 338.3595 338.5752
## [57] 338.7435 338.8831 338.9876 339.1473 339.2906 339.3939 339.5482
## [64] 339.6629 339.7905 339.9023 340.0149 340.1043 340.1895 340.2630
## [71] 340.3166 340.4055 340.4749 340.4959 340.5864 340.6462
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.6756
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 356.5235
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
## 1  255.2820965 203502.7  22813.73  226316.5 180689.00     0
## 2  232.6035386 194611.1  22218.33  216829.4 172392.74     1
## 3  211.9396813 185750.1  21170.01  206920.1 164580.08     2
## 4  193.1115442 178664.9  20361.05  199025.9 158303.81     2
## 5  175.9560468 172280.2  19724.49  192004.7 152555.67     3
## 6  160.3245966 165653.0  19230.79  184883.7 146422.18     4
## 7  146.0818013 159394.8  18873.55  178268.3 140521.22     4
## 8  133.1042967 153856.9  18580.45  172437.3 135276.43     4
## 9  121.2796778 149199.2  18355.28  167554.5 130843.95     4
## 10 110.5055255 145275.8  18194.48  163470.3 127081.30     4
## 11 100.6885192 142086.9  18049.53  160136.4 124037.35     5
## 12  91.7436287 139368.7  17912.42  157281.1 121456.25     5
## 13  83.5933775 137129.6  17792.19  154921.7 119337.37     5
## 14  76.1671723 134951.1  17649.12  152600.2 117301.97     5
## 15  69.4006906 132883.2  17494.61  150377.8 115388.57     6
## 16  63.2353245 130707.6  17335.69  148043.3 113371.87     6
## 17  57.6176726 128738.0  17187.18  145925.2 111550.79     6
## 18  52.4990774 127109.0  17065.08  144174.1 110043.90     6
## 19  47.8352040 125716.8  16937.10  142653.9 108779.73     6
## 20  43.5856563 124476.7  16777.36  141254.0 107699.32     6
## 21  39.7136268 123424.8  16620.49  140045.3 106804.28     6
## 22  36.1855776 122558.2  16486.20  139044.4 106071.97     6
## 23  32.9709506 121840.7  16370.98  138211.6 105469.68     6
## 24  30.0419022 121222.2  16253.07  137475.3 104969.13     6
## 25  27.3730624 120698.7  16140.35  136839.0 104558.33     6
## 26  24.9413150 120263.4  16044.27  136307.6 104219.08     6
## 27  22.7255973 119947.7  15968.37  135916.1 103979.32     6
## 28  20.7067179 119747.4  15895.64  135643.1 103851.80     6
## 29  18.8671902 119599.4  15831.92  135431.3 103767.48     6
## 30  17.1910810 119499.0  15774.82  135273.8 103724.19     7
## 31  15.6638727 119435.1  15726.52  135161.6 103708.54     7
## 32  14.2723374 119400.3  15684.19  135084.4 103716.06     7
## 33  13.0044223 119391.2  15646.77  135037.9 103744.39     9
## 34  11.8491453 119373.3  15615.34  134988.7 103758.00     9
## 35  10.7964999 119373.7  15588.19  134961.9 103785.50     9
## 36   9.8373686 119398.8  15563.77  134962.5 103835.00     9
## 37   8.9634439 119459.1  15515.35  134974.5 103943.78     9
## 38   8.1671562 119494.0  15421.03  134915.0 104072.98    11
## 39   7.4416086 119380.7  15354.80  134735.5 104025.87    11
## 40   6.7805166 119095.4  15307.67  134403.0 103787.68    12
## 41   6.1781542 118431.9  15224.08  133656.0 103207.85    12
## 42   5.6293040 117611.1  15084.02  132695.2 102527.11    13
## 43   5.1292121 116779.2  14974.98  131754.2 101804.21    13
## 44   4.6735471 116084.6  14868.97  130953.5 101215.59    13
## 45   4.2583620 115512.3  14776.47  130288.8 100735.84    13
## 46   3.8800609 115057.9  14698.42  129756.4 100359.52    13
## 47   3.5353670 114684.2  14608.70  129292.9 100075.47    13
## 48   3.2212947 114348.2  14507.09  128855.3  99841.10    13
## 49   2.9351238 114134.1  14427.42  128561.5  99706.64    13
## 50   2.6743755 114026.0  14371.62  128397.6  99654.38    13
## 51   2.4367913 114024.8  14312.57  128337.4  99712.27    13
## 52   2.2203135 114115.6  14264.65  128380.3  99850.96    14
## 53   2.0230670 114216.1  14223.63  128439.8  99992.50    15
## 54   1.8433433 114354.0  14184.15  128538.1 100169.83    15
## 55   1.6795857 114487.2  14155.42  128642.6 100331.75    17
## 56   1.5303760 114633.2  14121.51  128754.7 100511.66    17
## 57   1.3944216 114747.1  14089.88  128837.0 100657.25    17
## 58   1.2705450 114841.8  14076.38  128918.1 100765.39    17
## 59   1.1576733 114912.6  14073.29  128985.9 100839.32    17
## 60   1.0548288 115020.9  14069.93  129090.8 100950.93    17
## 61   0.9611207 115118.1  14062.83  129180.9 101055.26    17
## 62   0.8757374 115188.2  14055.38  129243.6 101132.80    17
## 63   0.7979393 115293.0  14050.06  129343.1 101242.94    17
## 64   0.7270526 115370.9  14049.06  129420.0 101321.84    17
## 65   0.6624632 115457.6  14049.60  129507.2 101408.01    18
## 66   0.6036118 115533.6  14050.40  129584.0 101483.16    18
## 67   0.5499886 115610.1  14052.59  129662.7 101557.53    18
## 68   0.5011291 115670.9  14058.68  129729.6 101612.22    17
## 69   0.4566102 115728.9  14062.61  129791.5 101666.31    18
## 70   0.4160462 115778.9  14060.26  129839.2 101718.63    18
## 71   0.3790858 115815.4  14061.98  129877.4 101753.40    18
## 72   0.3454089 115875.9  14064.82  129940.7 101811.07    18
## 73   0.3147237 115923.1  14065.46  129988.6 101857.66    18
## 74   0.2867645 115937.4  14070.16  130007.6 101867.27    18
## 75   0.2612891 115999.1  14070.38  130069.5 101928.74    18
## 76   0.2380769 116039.9  14069.41  130109.3 101970.45    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   52.49908
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

![](14-shrink_files/figure-latex/unnamed-chunk-15-1.pdf)<!-- --> 


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

![](14-shrink_files/figure-latex/unnamed-chunk-19-1.pdf)<!-- --> 


```r
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```

![](14-shrink_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 

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
