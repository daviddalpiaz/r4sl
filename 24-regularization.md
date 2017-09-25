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
## (Intercept) 268.287904048
## AtBat         0.075738253
## Hits          0.300154606
## HmRun         1.022784256
## Runs          0.489474365
## RBI           0.495632199
## Walks         0.626356706
## Years         2.143185629
## CAtBat        0.006369369
## CHits         0.024201921
## CHmRun        0.180499284
## CRuns         0.048544437
## CRBI          0.050169414
## CWalks        0.049897906
## LeagueN       1.802540422
## DivisionW   -16.185025138
## PutOuts       0.040146198
## Assists       0.005930000
## Errors       -0.087618226
## NewLeagueN    1.836629079
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
## (Intercept) 268.287904048
## AtBat         0.075738253
## Hits          0.300154606
## HmRun         1.022784256
## Runs          0.489474365
## RBI           0.495632199
## Walks         0.626356706
## Years         2.143185629
## CAtBat        0.006369369
## CHits         0.024201921
## CHmRun        0.180499284
## CRuns         0.048544437
## CRBI          0.050169414
## CWalks        0.049897906
## LeagueN       1.802540422
## DivisionW   -16.185025138
## PutOuts       0.040146198
## Assists       0.005930000
## Errors       -0.087618226
## NewLeagueN    1.836629079
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 275.24
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 141009.7
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 452.2600 450.9807 449.9086 449.6476 449.3624 449.0508 448.7104
##  [8] 448.3389 447.9336 447.4917 447.0100 446.4856 445.9148 445.2942
## [15] 444.6200 443.8882 443.0947 442.2355 441.3061 440.3022 439.2193
## [22] 438.0531 436.7994 435.4540 434.0132 432.4735 430.8320 429.0863
## [29] 427.2350 425.2772 423.2133 421.0449 418.7746 416.4067 413.9469
## [36] 411.4025 408.7825 406.0972 403.3585 400.5798 397.7758 394.9618
## [43] 392.1543 389.3696 386.6244 383.9346 381.3157 378.7817 376.3452
## [50] 374.0170 371.8059 369.7186 367.7599 365.9329 364.2377 362.6734
## [57] 361.2370 359.9244 358.7304 357.6488 356.6729 355.7958 355.0104
## [64] 354.3093 353.6869 353.1377 352.6517 352.2246 351.8486 351.5223
## [71] 351.2431 351.0002 350.7890 350.6095 350.4593 350.3288 350.2147
## [78] 350.1222 350.0377 349.9682 349.9007 349.8408 349.7829 349.7264
## [85] 349.6671 349.6054 349.5376 349.4654 349.3880 349.3014 349.2069
## [92] 349.1049 348.9937 348.8736 348.7460 348.6109 348.4694 348.3219
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 348.3219
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 383.9346
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
##  [1] 448.9093 441.0495 431.1581 422.9619 415.6455 407.8336 400.1123
##  [8] 393.1318 386.8953 381.8222 377.7798 374.2998 371.1347 368.1619
## [15] 365.2099 362.1258 359.2647 356.8184 354.7977 353.1306 351.7559
## [22] 350.6391 349.7175 349.0005 348.5142 348.2145 348.0343 347.9534
## [29] 347.9608 348.0419 348.1634 348.4621 348.9538 349.4520 349.9789
## [36] 350.6300 351.3800 352.0162 351.9314 351.3985 350.4470 349.5457
## [43] 348.6222 347.7878 347.0941 346.5357 346.0952 345.7480 345.4626
## [50] 345.4124 345.4737 345.6785 345.9423 346.2570 346.5099 346.8022
## [57] 347.1100 347.3828 347.7011 348.0022 348.2333 348.4642 348.6992
## [64] 348.8857 349.0571 349.2133 349.3549 349.4629 349.5697 349.7197
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 345.4124
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.1347
```

## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
#fit_lasso_cv
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high conf.low nzero
## 1  255.2820965 201519.6  19666.72  221186.3 181852.9     0
## 2  232.6035386 194524.7  20280.03  214804.7 174244.6     1
## 3  211.9396813 185897.4  19117.79  205015.1 166779.6     2
## 4  193.1115442 178896.8  18078.76  196975.6 160818.0     2
## 5  175.9560468 172761.1  17345.66  190106.8 155415.5     3
## 6  160.3245966 166328.3  16853.75  183182.0 149474.5     4
## 7  146.0818013 160089.9  16625.15  176715.0 143464.7     4
## 8  133.1042967 154552.6  16523.48  171076.1 138029.1     4
## 9  121.2796778 149687.9  16496.36  166184.3 133191.6     4
## 10 110.5055255 145788.2  16532.88  162321.1 129255.3     4
## 11 100.6885192 142717.6  16625.67  159343.3 126091.9     5
## 12  91.7436287 140100.4  16793.19  156893.6 123307.2     5
## 13  83.5933775 137740.9  17042.24  154783.2 120698.7     5
## 14  76.1671723 135543.2  17291.56  152834.8 118251.7     5
## 15  69.4006906 133378.3  17454.15  150832.5 115924.1     6
## 16  63.2353245 131135.1  17549.49  148684.6 113585.6     6
## 17  57.6176726 129071.1  17579.44  146650.5 111491.7     6
## 18  52.4990774 127319.3  17586.20  144905.5 109733.2     6
## 19  47.8352040 125881.4  17594.96  143476.3 108286.4     6
## 20  43.5856563 124701.2  17604.85  142306.1 107096.4     6
## 21  39.7136268 123732.2  17615.34  141347.6 106116.9     6
## 22  36.1855776 122947.8  17624.54  140572.3 105323.2     6
## 23  32.9709506 122302.3  17633.51  139935.9 104668.8     6
## 24  30.0419022 121801.4  17653.35  139454.7 104148.0     6
## 25  27.3730624 121462.2  17697.81  139160.0 103764.4     6
## 26  24.9413150 121253.4  17759.75  139013.1 103493.6     6
## 27  22.7255973 121127.9  17826.02  138953.9 103301.9     6
## 28  20.7067179 121071.6  17891.84  138963.4 103179.7     6
## 29  18.8671902 121076.7  17957.11  139033.8 103119.6     6
## 30  17.1910810 121133.2  18013.25  139146.4 103119.9     7
## 31  15.6638727 121217.8  18068.23  139286.0 103149.5     7
## 32  14.2723374 121425.8  18175.49  139601.3 103250.3     7
## 33  13.0044223 121768.8  18342.78  140111.5 103426.0     9
## 34  11.8491453 122116.7  18523.92  140640.6 103592.8     9
## 35  10.7964999 122485.3  18702.49  141187.8 103782.8     9
## 36   9.8373686 122941.4  18870.10  141811.5 104071.3     9
## 37   8.9634439 123467.9  19013.49  142481.4 104454.4     9
## 38   8.1671562 123915.4  19128.23  143043.6 104787.2    11
## 39   7.4416086 123855.7  19129.16  142984.9 104726.6    11
## 40   6.7805166 123480.9  19042.99  142523.9 104437.9    12
## 41   6.1781542 122813.1  19002.60  141815.7 103810.5    12
## 42   5.6293040 122182.2  18942.04  141124.2 103240.1    13
## 43   5.1292121 121537.4  18903.58  140441.0 102633.8    13
## 44   4.6735471 120956.3  18890.57  139846.9 102065.8    13
## 45   4.2583620 120474.3  18860.04  139334.4 101614.3    13
## 46   3.8800609 120087.0  18818.58  138905.6 101268.4    13
## 47   3.5353670 119781.9  18782.62  138564.5 100999.3    13
## 48   3.2212947 119541.7  18746.87  138288.5 100794.8    13
## 49   2.9351238 119344.4  18709.06  138053.5 100635.4    13
## 50   2.6743755 119309.8  18678.15  137987.9 100631.6    13
## 51   2.4367913 119352.1  18646.74  137998.8 100705.3    13
## 52   2.2203135 119493.6  18597.78  138091.4 100895.9    14
## 53   2.0230670 119676.0  18571.19  138247.2 101104.9    15
## 54   1.8433433 119893.9  18556.45  138450.3 101337.4    15
## 55   1.6795857 120069.1  18554.48  138623.6 101514.6    17
## 56   1.5303760 120271.8  18558.71  138830.5 101713.1    17
## 57   1.3944216 120485.3  18567.81  139053.2 101917.5    17
## 58   1.2705450 120674.8  18573.38  139248.2 102101.5    17
## 59   1.1576733 120896.0  18587.17  139483.2 102308.9    17
## 60   1.0548288 121105.5  18607.18  139712.7 102498.3    17
## 61   0.9611207 121266.4  18632.43  139898.8 102634.0    17
## 62   0.8757374 121427.3  18648.85  140076.1 102778.4    17
## 63   0.7979393 121591.1  18657.28  140248.4 102933.8    17
## 64   0.7270526 121721.3  18666.95  140388.2 103054.3    17
## 65   0.6624632 121840.9  18675.38  140516.3 103165.5    18
## 66   0.6036118 121949.9  18683.78  140633.7 103266.1    18
## 67   0.5499886 122048.8  18694.27  140743.1 103354.6    18
## 68   0.5011291 122124.3  18701.06  140825.4 103423.3    17
## 69   0.4566102 122198.9  18712.63  140911.6 103486.3    18
## 70   0.4160462 122303.9  18711.95  141015.9 103591.9    18
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
