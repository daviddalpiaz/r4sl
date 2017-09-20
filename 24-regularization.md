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
## (Intercept) 226.844379273
## AtBat         0.086613903
## Hits          0.352962516
## HmRun         1.144213853
## Runs          0.569353374
## RBI           0.570074068
## Walks         0.735072620
## Years         2.397356093
## CAtBat        0.007295083
## CHits         0.027995153
## CHmRun        0.208112350
## CRuns         0.056146220
## CRBI          0.058060281
## CWalks        0.056586702
## LeagueN       2.850306112
## DivisionW   -20.329125702
## PutOuts       0.049296951
## Assists       0.007063169
## Errors       -0.128066381
## NewLeagueN    2.654025563
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
## (Intercept) 226.844379273
## AtBat         0.086613903
## Hits          0.352962516
## HmRun         1.144213853
## Runs          0.569353374
## RBI           0.570074068
## Walks         0.735072620
## Years         2.397356093
## CAtBat        0.007295083
## CHits         0.027995153
## CHmRun        0.208112350
## CRuns         0.056146220
## CRBI          0.058060281
## CWalks        0.056586702
## LeagueN       2.850306112
## DivisionW   -20.329125702
## PutOuts       0.049296951
## Assists       0.007063169
## Errors       -0.128066381
## NewLeagueN    2.654025563
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 436.8923
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 134397.5
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 450.9932 449.1487 448.5222 448.2512 447.9550 447.6313 447.2778
##  [8] 446.8918 446.4706 446.0112 445.5105 444.9650 444.3714 443.7256
## [15] 443.0239 442.2619 441.4355 440.5401 439.5712 438.5240 437.3939
## [22] 436.1761 434.8660 433.4592 431.9513 430.3387 428.6178 426.7859
## [29] 424.8409 422.7817 420.6081 418.3211 415.9232 413.4182 410.8113
## [36] 408.1098 405.3223 402.4590 399.5321 396.5550 393.5428 390.5115
## [43] 387.4780 384.4599 381.4745 378.5394 375.6712 372.8855 370.1966
## [50] 367.6169 365.1566 362.8248 360.6277 358.5695 356.6521 354.8756
## [57] 353.2383 351.7368 350.3666 349.1219 347.9963 346.9829 346.0745
## [64] 345.2639 344.5459 343.9126 343.3552 342.8679 342.4449 342.0827
## [71] 341.7738 341.5116 341.2919 341.1125 340.9666 340.8491 340.7595
## [78] 340.6908 340.6414 340.6065 340.5845 340.5723 340.5673 340.5658
## [85] 340.5682 340.5720 340.5734 340.5730 340.5713 340.5643 340.5530
## [92] 340.5373 340.5169 340.4917 340.4617 340.4287 340.3924 340.3529
## [99] 340.3126
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.3126
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.1966
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
## (Intercept)   94.9413141
## AtBat         -1.2097544
## Hits           4.9149788
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.1260566
## Years         -7.7221755
## CAtBat         .        
## CHits          .        
## CHmRun         0.3722512
## CRuns          0.5486528
## CRBI           0.4055863
## CWalks        -0.3912414
## LeagueN       30.0322424
## DivisionW   -119.0746672
## PutOuts        0.2632822
## Assists        0.1098523
## Errors        -1.5222432
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 170.693
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
##  [1] 450.1248 440.2912 430.2100 421.6521 413.4347 404.8815 397.1991
##  [8] 390.3529 384.2601 379.0257 374.7282 370.9158 367.5868 364.4362
## [15] 361.2223 358.3149 355.6122 353.1877 351.1261 349.4144 348.0051
## [22] 346.8447 345.8950 345.1164 344.4816 343.9880 343.6314 343.4208
## [29] 343.2913 343.2256 343.1993 343.1750 343.1712 343.1114 343.0914
## [36] 343.1091 343.1857 343.2929 343.3349 342.9244 342.1917 341.5918
## [43] 341.1184 340.8201 340.6601 340.5657 340.6095 340.7051 340.8605
## [50] 340.6980 340.6550 340.6715 340.7339 341.0223 340.9740 341.1528
## [57] 341.2903 341.3493 341.3157 341.3213 341.3449 341.4194 341.4962
## [64] 341.5877 341.7421 341.8585 342.0075 342.1206 342.2608 342.3480
## [71] 342.4384 342.5126 342.5827
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.5657
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.5868
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
## 1  255.2820965 202612.3  26490.92  229103.2 176121.39     0
## 2  232.6035386 193856.3  25486.15  219342.5 168370.17     1
## 3  211.9396813 185080.7  24293.23  209373.9 160787.45     2
## 4  193.1115442 177790.5  23250.66  201041.2 154539.84     2
## 5  175.9560468 170928.3  22438.24  193366.5 148490.04     3
## 6  160.3245966 163929.0  21844.52  185773.5 142084.52     4
## 7  146.0818013 157767.1  21375.52  179142.7 136391.62     4
## 8  133.1042967 152375.4  21034.95  173410.3 131340.44     4
## 9  121.2796778 147655.8  20748.00  168403.8 126907.81     4
## 10 110.5055255 143660.5  20536.14  164196.6 123124.34     4
## 11 100.6885192 140421.2  20390.85  160812.1 120030.34     5
## 12  91.7436287 137578.5  20171.93  157750.5 117406.62     5
## 13  83.5933775 135120.1  20012.36  155132.4 115107.71     5
## 14  76.1671723 132813.8  19973.27  152787.1 112840.51     5
## 15  69.4006906 130481.6  20036.41  150518.0 110445.17     6
## 16  63.2353245 128389.6  20159.21  148548.8 108230.37     6
## 17  57.6176726 126460.1  20252.82  146712.9 106207.23     6
## 18  52.4990774 124741.6  20258.45  145000.0 104483.11     6
## 19  47.8352040 123289.5  20258.59  143548.1 103030.94     6
## 20  43.5856563 122090.5  20279.44  142369.9 101811.01     6
## 21  39.7136268 121107.5  20314.60  141422.1 100792.94     6
## 22  36.1855776 120301.3  20360.57  140661.8  99940.70     6
## 23  32.9709506 119643.3  20413.57  140056.9  99229.77     6
## 24  30.0419022 119105.3  20470.34  139575.7  98634.98     6
## 25  27.3730624 118667.6  20531.99  139199.6  98135.57     6
## 26  24.9413150 118327.7  20594.98  138922.7  97732.77     6
## 27  22.7255973 118082.5  20652.43  138734.9  97430.09     6
## 28  20.7067179 117937.8  20706.64  138644.5  97231.18     6
## 29  18.8671902 117848.9  20757.84  138606.8  97091.07     6
## 30  17.1910810 117803.8  20822.62  138626.4  96981.17     7
## 31  15.6638727 117785.7  20895.96  138681.7  96889.78     7
## 32  14.2723374 117769.1  20957.39  138726.5  96811.68     7
## 33  13.0044223 117766.5  21011.54  138778.0  96754.95     9
## 34  11.8491453 117725.4  21063.85  138789.3  96661.57     9
## 35  10.7964999 117711.7  21118.05  138829.8  96593.66     9
## 36   9.8373686 117723.8  21161.77  138885.6  96562.06     9
## 37   8.9634439 117776.4  21193.58  138970.0  96582.86     9
## 38   8.1671562 117850.0  21243.77  139093.8  96606.27    11
## 39   7.4416086 117878.9  21320.75  139199.6  96558.11    11
## 40   6.7805166 117597.2  21240.63  138837.8  96356.52    12
## 41   6.1781542 117095.2  21090.89  138186.1  96004.30    12
## 42   5.6293040 116685.0  20955.49  137640.5  95729.48    13
## 43   5.1292121 116361.8  20857.44  137219.2  95504.35    13
## 44   4.6735471 116158.4  20798.91  136957.3  95359.45    13
## 45   4.2583620 116049.3  20761.71  136811.0  95287.56    13
## 46   3.8800609 115985.0  20705.76  136690.7  95279.22    13
## 47   3.5353670 116014.8  20633.55  136648.4  95381.26    13
## 48   3.2212947 116080.0  20568.80  136648.8  95511.19    13
## 49   2.9351238 116185.9  20475.24  136661.1  95710.66    13
## 50   2.6743755 116075.1  20286.96  136362.1  95788.14    13
## 51   2.4367913 116045.8  20135.68  136181.5  95910.16    13
## 52   2.2203135 116057.1  20005.32  136062.4  96051.77    14
## 53   2.0230670 116099.6  19886.29  135985.9  96213.32    15
## 54   1.8433433 116296.2  19804.66  136100.9  96491.53    15
## 55   1.6795857 116263.3  19669.97  135933.2  96593.29    17
## 56   1.5303760 116385.3  19591.81  135977.1  96793.46    17
## 57   1.3944216 116479.1  19515.88  135995.0  96963.20    17
## 58   1.2705450 116519.4  19420.06  135939.4  97099.31    17
## 59   1.1576733 116496.4  19312.45  135808.9  97183.96    17
## 60   1.0548288 116500.2  19235.18  135735.4  97265.03    17
## 61   0.9611207 116516.4  19166.74  135683.1  97349.63    17
## 62   0.8757374 116567.2  19096.49  135663.7  97470.69    17
## 63   0.7979393 116619.6  19033.63  135653.3  97586.02    17
## 64   0.7270526 116682.2  18974.72  135656.9  97707.46    17
## 65   0.6624632 116787.6  18923.26  135710.9  97864.39    18
## 66   0.6036118 116867.2  18876.55  135743.8  97990.67    18
## 67   0.5499886 116969.2  18831.96  135801.1  98137.19    18
## 68   0.5011291 117046.5  18792.84  135839.4  98253.69    17
## 69   0.4566102 117142.4  18753.51  135895.9  98388.91    18
## 70   0.4160462 117202.1  18721.31  135923.4  98480.82    18
## 71   0.3790858 117264.0  18688.97  135953.0  98575.07    18
## 72   0.3454089 117314.9  18665.93  135980.8  98648.95    18
## 73   0.3147237 117362.9  18642.08  136005.0  98720.83    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   3.880061   83.59338
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
