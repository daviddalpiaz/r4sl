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
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
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
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 588.9958
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 130404.9
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 452.6726 450.9391 450.1903 449.9051 449.6131 449.2941 448.9456
##  [8] 448.5652 448.1500 447.6971 447.2035 446.6658 446.0804 445.4437
## [15] 444.7517 444.0001 443.1850 442.3017 441.3458 440.3125 439.1971
## [22] 437.9950 436.7016 435.3122 433.8228 432.2294 430.5285 428.7173
## [29] 426.7935 424.7559 422.6040 420.3388 417.9621 415.4776 412.8902
## [36] 410.2066 407.4350 404.5850 401.6682 398.6975 395.6873 392.6530
## [43] 389.6109 386.5778 383.5706 380.6063 377.7010 374.8700 372.1275
## [50] 369.4854 366.9540 364.5419 362.2563 360.1010 358.0785 356.1892
## [57] 354.4318 352.8034 351.2998 349.9161 348.6461 347.4835 346.4214
## [64] 345.4519 344.5701 343.7745 343.0491 342.3913 341.7952 341.2548
## [71] 340.7668 340.3303 339.9300 339.5656 339.2353 338.9374 338.6598
## [78] 338.4057 338.1702 337.9493 337.7421 337.5433 337.3540 337.1696
## [85] 336.9882 336.8121 336.6358 336.4600 336.2863 336.1104 335.9350
## [92] 335.7594 335.5843 335.4094 335.2374 335.0663 334.9009 334.7386
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 334.7386
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 366.954
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
##  [1] 449.9307 442.9919 433.2029 424.8754 417.2031 408.9250 400.4252
##  [8] 392.8631 386.5132 381.4362 377.1640 373.4117 370.1329 367.1234
## [15] 364.1665 361.2233 358.4007 355.8567 353.7426 352.0020 350.5559
## [22] 349.3668 348.3897 347.5877 346.9295 346.3977 345.9814 345.6791
## [29] 345.4508 345.3216 345.2691 345.2284 345.2171 345.2137 345.2521
## [36] 345.3973 345.7566 346.0670 345.9731 345.3085 344.5237 343.9283
## [43] 343.1690 342.4164 341.7007 341.1311 340.6825 340.3926 340.1705
## [50] 340.0047 339.9070 339.9616 340.1384 340.3135 340.3649 340.4186
## [57] 340.4973 340.5680 340.5904 340.6510 340.7369 340.8478 340.9621
## [64] 341.0599 341.1511 341.2589 341.3545 341.4358 341.5216 341.5715
## [71] 341.6529 341.7065 341.7727 341.8145
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.907
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.1329
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
## 1  255.2820965 202437.6  27561.22  229998.9 174876.43     0
## 2  232.6035386 196241.8  27697.64  223939.5 168544.19     1
## 3  211.9396813 187664.8  26224.60  213889.4 161440.17     2
## 4  193.1115442 180519.1  24965.68  205484.8 155553.45     2
## 5  175.9560468 174058.4  23969.70  198028.1 150088.73     3
## 6  160.3245966 167219.7  22937.41  190157.1 144282.27     4
## 7  146.0818013 160340.3  21929.46  182269.8 138410.85     4
## 8  133.1042967 154341.4  21151.66  175493.1 133189.73     4
## 9  121.2796778 149392.4  20581.46  169973.9 128810.96     4
## 10 110.5055255 145493.6  20194.45  165688.0 125299.12     4
## 11 100.6885192 142252.7  19916.78  162169.5 122335.91     5
## 12  91.7436287 139436.3  19730.32  159166.6 119705.98     5
## 13  83.5933775 136998.4  19601.56  156600.0 117396.84     5
## 14  76.1671723 134779.6  19561.06  154340.7 115218.53     5
## 15  69.4006906 132617.2  19593.50  152210.7 113023.74     6
## 16  63.2353245 130482.3  19645.67  150128.0 110836.63     6
## 17  57.6176726 128451.1  19693.08  148144.2 108758.01     6
## 18  52.4990774 126634.0  19706.18  146340.2 106927.83     6
## 19  47.8352040 125133.8  19761.55  144895.4 105372.27     6
## 20  43.5856563 123905.4  19845.11  143750.5 104060.33     6
## 21  39.7136268 122889.4  19949.25  142838.7 102940.17     6
## 22  36.1855776 122057.2  20066.52  142123.7 101990.66     6
## 23  32.9709506 121375.4  20190.46  141565.8 101184.92     6
## 24  30.0419022 120817.2  20317.09  141134.3 100500.11     6
## 25  27.3730624 120360.1  20443.72  140803.8  99916.37     6
## 26  24.9413150 119991.4  20566.86  140558.3  99424.54     6
## 27  22.7255973 119703.1  20687.09  140390.2  99016.04     6
## 28  20.7067179 119494.0  20799.05  140293.1  98695.00     6
## 29  18.8671902 119336.3  20896.11  140232.4  98440.17     6
## 30  17.1910810 119247.0  20987.80  140234.8  98259.22     7
## 31  15.6638727 119210.8  21070.29  140281.1  98140.48     7
## 32  14.2723374 119182.6  21156.48  140339.1  98026.13     7
## 33  13.0044223 119174.9  21255.45  140430.3  97919.40     9
## 34  11.8491453 119172.5  21352.44  140524.9  97820.05     9
## 35  10.7964999 119199.0  21454.31  140653.3  97744.70     9
## 36   9.8373686 119299.3  21614.81  140914.1  97684.47     9
## 37   8.9634439 119547.6  21851.85  141399.4  97695.75     9
## 38   8.1671562 119762.4  22067.65  141830.0  97694.71    11
## 39   7.4416086 119697.4  22069.49  141766.9  97627.93    11
## 40   6.7805166 119237.9  22000.28  141238.2  97237.66    12
## 41   6.1781542 118696.6  21957.98  140654.6  96738.62    12
## 42   5.6293040 118286.7  21916.37  140203.1  96370.32    13
## 43   5.1292121 117764.9  21883.01  139647.9  95881.94    13
## 44   4.6735471 117249.0  21842.66  139091.6  95406.33    13
## 45   4.2583620 116759.4  21810.50  138569.9  94948.88    13
## 46   3.8800609 116370.4  21810.42  138180.8  94560.01    13
## 47   3.5353670 116064.5  21830.53  137895.1  94234.02    13
## 48   3.2212947 115867.1  21853.59  137720.7  94013.54    13
## 49   2.9351238 115716.0  21886.29  137602.3  93829.68    13
## 50   2.6743755 115603.2  21916.00  137519.2  93687.23    13
## 51   2.4367913 115536.8  21952.91  137489.7  93583.88    13
## 52   2.2203135 115573.9  22031.63  137605.5  93542.25    14
## 53   2.0230670 115694.1  22156.06  137850.2  93538.06    15
## 54   1.8433433 115813.3  22271.63  138084.9  93541.64    15
## 55   1.6795857 115848.2  22390.16  138238.4  93458.07    17
## 56   1.5303760 115884.9  22501.75  138386.6  93383.11    17
## 57   1.3944216 115938.4  22602.77  138541.2  93335.64    17
## 58   1.2705450 115986.6  22695.69  138682.3  93290.90    17
## 59   1.1576733 116001.8  22788.95  138790.8  93212.88    17
## 60   1.0548288 116043.1  22870.35  138913.5  93172.76    17
## 61   0.9611207 116101.6  22928.23  139029.9  93173.40    17
## 62   0.8757374 116177.2  22977.10  139154.3  93200.13    17
## 63   0.7979393 116255.2  23017.43  139272.6  93237.73    17
## 64   0.7270526 116321.8  23048.67  139370.5  93273.17    17
## 65   0.6624632 116384.1  23072.31  139456.4  93311.75    18
## 66   0.6036118 116457.6  23086.41  139544.0  93371.21    18
## 67   0.5499886 116522.9  23096.89  139619.8  93426.02    18
## 68   0.5011291 116578.4  23105.55  139684.0  93472.88    17
## 69   0.4566102 116637.0  23112.87  139749.9  93524.15    18
## 70   0.4160462 116671.1  23118.77  139789.9  93552.35    18
## 71   0.3790858 116726.7  23125.26  139852.0  93601.47    18
## 72   0.3454089 116763.3  23129.64  139893.0  93633.67    18
## 73   0.3147237 116808.6  23134.80  139943.4  93673.77    18
## 74   0.2867645 116837.2  23139.23  139976.4  93697.95    18
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
