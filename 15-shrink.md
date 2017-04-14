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
##  [1] 450.9642 449.4066 448.5964 448.2907 447.9988 447.6798 447.3314
##  [8] 446.9511 446.5361 446.0835 445.5902 445.0529 444.4680 443.8320
## [15] 443.1408 442.3903 441.5765 440.6948 439.7408 438.7099 437.5974
## [22] 436.3988 435.1095 433.7251 432.2416 430.6552 428.9626 427.1611
## [29] 425.2488 423.2246 421.0883 418.8411 416.4854 414.0248 411.4648
## [36] 408.8124 406.0761 403.2658 400.3935 397.4723 394.5168 391.5427
## [43] 388.5663 385.6044 382.6742 379.7922 376.9745 374.2360 371.5903
## [50] 369.0493 366.6227 364.3180 362.1422 360.0985 358.1885 356.4122
## [57] 354.7675 353.2511 351.8583 350.5834 349.4203 348.3621 347.4017
## [64] 346.5314 345.7459 345.0379 344.4003 343.8272 343.3113 342.8495
## [71] 342.4380 342.0696 341.7342 341.4366 341.1662 340.9194 340.6970
## [78] 340.4908 340.3019 340.1215 339.9531 339.7891 339.6305 339.4777
## [85] 339.3164 339.1668 339.0011 338.8476 338.6740 338.5189 338.3311
## [92] 338.1711 337.9724 337.8035 337.6027 337.4300 337.2270 337.0580
## [99] 336.8551
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.8551
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 362.1422
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
##  [1] 449.0010 441.2882 431.1016 422.4678 414.1976 405.4629 397.0774
##  [8] 389.8139 383.6651 378.5844 374.3342 370.6879 367.4678 364.6382
## [15] 361.7478 358.7626 355.9301 353.4145 351.2520 349.3934 347.8241
## [22] 346.5098 345.4739 344.7182 344.1022 343.5903 343.2378 342.9828
## [29] 342.8293 342.7440 342.7096 342.7089 342.7290 342.7486 342.7486
## [36] 342.7731 342.8364 342.6341 342.3596 341.7130 340.6936 339.5814
## [43] 338.5607 337.5617 336.7048 336.0473 335.5515 335.1831 334.9055
## [50] 334.7009 334.6031 334.6522 334.7674 334.8751 334.9518 334.9189
## [57] 334.8930 334.9133 334.9500 335.0037 335.1069 335.2298 335.3857
## [64] 335.5358 335.6759 335.8243 335.9591 336.1067 336.2542 336.3776
## [71] 336.5338 336.6226 336.7514 336.7995
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 334.6031
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 358.7626
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
## 1  255.2820965 201601.9  24376.67  225978.5 177225.19     0
## 2  232.6035386 194735.3  24021.51  218756.8 170713.77     1
## 3  211.9396813 185848.6  23244.14  209092.8 162604.48     2
## 4  193.1115442 178479.0  22664.04  201143.1 155814.99     2
## 5  175.9560468 171559.7  22302.03  193861.7 149257.64     3
## 6  160.3245966 164400.2  21904.00  186304.2 142496.20     4
## 7  146.0818013 157670.4  21509.16  179179.6 136161.28     4
## 8  133.1042967 151954.9  21216.73  173171.6 130738.13     4
## 9  121.2796778 147198.9  20988.59  168187.5 126210.32     4
## 10 110.5055255 143326.2  20800.86  164127.0 122525.29     4
## 11 100.6885192 140126.1  20656.50  160782.6 119469.62     5
## 12  91.7436287 137409.5  20577.22  157986.7 116832.28     5
## 13  83.5933775 135032.6  20520.55  155553.1 114512.03     5
## 14  76.1671723 132961.0  20408.81  153369.8 112552.19     5
## 15  69.4006906 130861.5  20311.31  151172.8 110550.17     6
## 16  63.2353245 128710.6  20252.65  148963.3 108457.97     6
## 17  57.6176726 126686.2  20155.84  146842.1 106530.38     6
## 18  52.4990774 124901.8  20027.90  144929.7 104873.92     6
## 19  47.8352040 123378.0  19914.18  143292.2 103463.80     6
## 20  43.5856563 122075.7  19812.70  141888.4 102263.02     6
## 21  39.7136268 120981.6  19728.55  140710.2 101253.08     6
## 22  36.1855776 120069.1  19660.15  139729.2 100408.90     6
## 23  32.9709506 119352.2  19615.51  138967.7  99736.68     6
## 24  30.0419022 118830.6  19599.24  138429.9  99231.39     6
## 25  27.3730624 118406.3  19587.98  137994.3  98818.36     6
## 26  24.9413150 118054.3  19574.02  137628.3  98480.28     6
## 27  22.7255973 117812.2  19554.65  137366.8  98257.54     6
## 28  20.7067179 117637.2  19536.81  137174.0  98100.39     6
## 29  18.8671902 117531.9  19515.88  137047.8  98016.04     6
## 30  17.1910810 117473.4  19497.27  136970.7  97976.14     7
## 31  15.6638727 117449.8  19479.70  136929.5  97970.15     7
## 32  14.2723374 117449.4  19463.73  136913.1  97985.64     7
## 33  13.0044223 117463.1  19450.18  136913.3  98012.96     9
## 34  11.8491453 117476.6  19436.51  136913.1  98040.11     9
## 35  10.7964999 117476.6  19415.24  136891.9  98061.38     9
## 36   9.8373686 117493.4  19380.29  136873.7  98113.13     9
## 37   8.9634439 117536.8  19338.14  136874.9  98198.66     9
## 38   8.1671562 117398.1  19310.61  136708.7  98087.51    11
## 39   7.4416086 117210.1  19283.14  136493.2  97926.96    11
## 40   6.7805166 116767.8  19199.47  135967.2  97568.29    12
## 41   6.1781542 116072.1  19010.30  135082.4  97061.81    12
## 42   5.6293040 115315.5  18824.03  134139.6  96491.49    13
## 43   5.1292121 114623.3  18636.27  133259.6  95987.06    13
## 44   4.6735471 113947.9  18422.87  132370.8  95525.04    13
## 45   4.2583620 113370.1  18213.09  131583.2  95157.02    13
## 46   3.8800609 112927.8  18022.70  130950.5  94905.11    13
## 47   3.5353670 112594.8  17851.66  130446.4  94743.12    13
## 48   3.2212947 112347.7  17695.06  130042.8  94652.66    13
## 49   2.9351238 112161.7  17553.77  129715.5  94607.95    13
## 50   2.6743755 112024.7  17416.47  129441.1  94608.19    13
## 51   2.4367913 111959.3  17280.39  129239.6  94678.86    13
## 52   2.2203135 111992.1  17130.37  129122.5  94861.74    14
## 53   2.0230670 112069.2  16986.41  129055.6  95082.80    15
## 54   1.8433433 112141.4  16853.12  128994.5  95288.24    15
## 55   1.6795857 112192.7  16714.11  128906.8  95478.63    17
## 56   1.5303760 112170.7  16572.94  128743.6  95597.75    17
## 57   1.3944216 112153.3  16458.95  128612.3  95694.36    17
## 58   1.2705450 112166.9  16367.96  128534.9  95798.97    17
## 59   1.1576733 112191.5  16305.26  128496.8  95886.27    17
## 60   1.0548288 112227.5  16264.01  128491.5  95963.47    17
## 61   0.9611207 112296.6  16223.62  128520.2  96072.99    17
## 62   0.8757374 112379.0  16185.00  128564.0  96194.01    17
## 63   0.7979393 112483.6  16145.47  128629.1  96338.13    17
## 64   0.7270526 112584.3  16100.21  128684.5  96484.09    17
## 65   0.6624632 112678.3  16049.28  128727.6  96629.06    18
## 66   0.6036118 112777.9  16003.50  128781.5  96774.44    18
## 67   0.5499886 112868.5  15963.61  128832.1  96904.92    18
## 68   0.5011291 112967.7  15927.13  128894.9  97040.62    17
## 69   0.4566102 113066.9  15890.36  128957.2  97176.52    18
## 70   0.4160462 113149.9  15864.16  129014.0  97285.70    18
## 71   0.3790858 113255.0  15836.12  129091.1  97418.90    18
## 72   0.3454089 113314.7  15808.79  129123.5  97505.96    18
## 73   0.3147237 113401.5  15788.48  129190.0  97613.00    18
## 74   0.2867645 113433.9  15765.24  129199.2  97668.68    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   63.23532
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
