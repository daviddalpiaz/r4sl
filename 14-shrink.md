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
##  [1] 451.4163 449.9373 449.0782 448.6892 448.3993 448.0825 447.7364
##  [8] 447.3586 446.9463 446.4966 446.0065 445.4726 444.8915 444.2595
## [15] 443.5726 442.8268 442.0178 441.1414 440.1930 439.1680 438.0618
## [22] 436.8698 435.5873 434.2102 432.7341 431.1553 429.4706 427.6770
## [29] 425.7726 423.7561 421.6274 419.3874 417.0383 414.5838 412.0289
## [36] 409.3804 406.6468 403.8376 400.9646 398.0406 395.0801 392.0984
## [43] 389.1117 386.1367 383.1901 380.2886 377.4481 374.6835 372.0084
## [50] 369.4351 366.9727 364.6300 362.4126 360.3252 358.3697 356.5460
## [57] 354.8527 353.2866 351.8434 350.5181 349.3045 348.1963 347.1866
## [64] 346.2672 345.4360 344.6845 344.0075 343.3953 342.8421 342.3469
## [71] 341.9024 341.5029 341.1467 340.8271 340.5387 340.2834 340.0494
## [78] 339.8401 339.6495 339.4757 339.3137 339.1659 339.0247 338.8896
## [85] 338.7624 338.6367 338.5153 338.3945 338.2750 338.1561 338.0370
## [92] 337.9159 337.7943 337.6748 337.5537 337.4331 337.3142 337.1983
## [99] 337.0835
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.0835
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 362.4126
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
## (Intercept)   20.9671790
## AtBat          .        
## Hits           1.8647356
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          2.2144706
## Years          .        
## CAtBat         .        
## CHits          .        
## CHmRun         .        
## CRuns          0.2067687
## CRBI           0.4123098
## CWalks         .        
## LeagueN        0.8359509
## DivisionW   -102.7759368
## PutOuts        0.2197627
## Assists        .        
## Errors         .        
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 108.5299
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
##  [1] 449.0782 441.6293 431.8624 423.6515 416.2300 407.9420 399.6051
##  [8] 392.1015 385.5607 380.0695 375.3355 370.8480 367.1472 364.0811
## [15] 361.3120 358.5010 355.7002 353.0936 350.9149 349.1032 347.6000
## [22] 346.3514 345.3108 344.4489 343.7911 343.3124 343.0486 342.9024
## [29] 342.8557 342.8512 342.9350 343.0732 343.4043 343.8062 344.1604
## [36] 344.7499 345.6905 346.4264 346.8707 346.7958 346.3962 345.9860
## [43] 345.5020 345.3090 345.1846 345.2236 345.4888 345.5189 345.6004
## [50] 345.8526 346.3856 347.0202 347.7267 348.4550 348.9865 349.4165
## [57] 349.7940 350.0870 350.3510 350.6064 350.8013 351.0101 351.2579
## [64] 351.5147 351.7648 351.9550 352.1634 352.3192 352.4794 352.6338
## [71] 352.7384 352.8114 352.9217 352.9785
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.8512
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.1472
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
## 1  255.2820965 201671.2  30511.46  232182.7 171159.76     0
## 2  232.6035386 195036.4  31023.53  226059.9 164012.88     1
## 3  211.9396813 186505.1  29792.23  216297.3 156712.90     2
## 4  193.1115442 179480.6  28650.00  208130.6 150830.58     2
## 5  175.9560468 173247.4  27712.06  200959.4 145535.31     3
## 6  160.3245966 166416.6  26870.66  193287.3 139545.99     4
## 7  146.0818013 159684.2  26005.49  185689.7 133678.71     4
## 8  133.1042967 153743.6  25220.50  178964.1 128523.11     4
## 9  121.2796778 148657.1  24483.12  173140.2 124173.96     4
## 10 110.5055255 144452.8  23847.90  168300.7 120604.92     4
## 11 100.6885192 140876.7  23309.80  164186.5 117566.91     5
## 12  91.7436287 137528.3  22796.31  160324.6 114731.95     5
## 13  83.5933775 134797.1  22392.92  157190.0 112404.18     5
## 14  76.1671723 132555.0  22030.82  154585.8 110524.20     5
## 15  69.4006906 130546.3  21716.54  152262.9 108829.80     6
## 16  63.2353245 128523.0  21391.28  149914.3 107131.69     6
## 17  57.6176726 126522.6  21004.80  147527.4 105517.80     6
## 18  52.4990774 124675.1  20604.34  145279.4 104070.76     6
## 19  47.8352040 123141.3  20264.94  143406.2 102876.35     6
## 20  43.5856563 121873.0  19974.23  141847.3 101898.81     6
## 21  39.7136268 120825.8  19723.32  140549.1 101102.47     6
## 22  36.1855776 119959.3  19508.30  139467.6 100450.98     6
## 23  32.9709506 119239.5  19324.12  138563.6  99915.39     6
## 24  30.0419022 118645.0  19164.99  137810.0  99480.06     6
## 25  27.3730624 118192.3  19024.78  137217.1  99167.53     6
## 26  24.9413150 117863.4  18904.64  136768.0  98958.74     6
## 27  22.7255973 117682.4  18812.85  136495.2  98869.50     6
## 28  20.7067179 117582.1  18735.03  136317.1  98847.03     6
## 29  18.8671902 117550.0  18676.22  136226.2  98873.80     6
## 30  17.1910810 117546.9  18624.96  136171.9  98921.99     7
## 31  15.6638727 117604.4  18583.20  136187.6  99021.18     7
## 32  14.2723374 117699.2  18548.04  136247.3  99151.17     7
## 33  13.0044223 117926.5  18516.96  136443.5  99409.56     9
## 34  11.8491453 118202.7  18494.97  136697.6  99707.70     9
## 35  10.7964999 118446.4  18489.05  136935.4  99957.32     9
## 36   9.8373686 118852.5  18477.02  137329.5 100375.51     9
## 37   8.9634439 119501.9  18457.25  137959.2 101044.67     9
## 38   8.1671562 120011.3  18425.86  138437.1 101585.43    11
## 39   7.4416086 120319.3  18341.01  138660.3 101978.27    11
## 40   6.7805166 120267.3  18219.02  138486.4 102048.32    12
## 41   6.1781542 119990.3  18066.27  138056.6 101924.07    12
## 42   5.6293040 119706.3  17933.49  137639.8 101772.83    13
## 43   5.1292121 119371.6  17750.90  137122.5 101620.70    13
## 44   4.6735471 119238.3  17622.52  136860.8 101615.77    13
## 45   4.2583620 119152.4  17491.95  136644.3 101660.43    13
## 46   3.8800609 119179.4  17389.53  136568.9 101789.83    13
## 47   3.5353670 119362.5  17354.86  136717.4 102007.64    13
## 48   3.2212947 119383.3  17360.07  136743.4 102023.22    13
## 49   2.9351238 119439.6  17366.75  136806.4 102072.87    13
## 50   2.6743755 119614.0  17386.48  137000.5 102227.51    13
## 51   2.4367913 119983.0  17428.62  137411.6 102554.37    13
## 52   2.2203135 120423.0  17484.61  137907.7 102938.44    14
## 53   2.0230670 120913.8  17563.93  138477.8 103349.91    15
## 54   1.8433433 121420.9  17652.02  139072.9 103768.86    15
## 55   1.6795857 121791.6  17713.49  139505.1 104078.12    17
## 56   1.5303760 122091.9  17797.48  139889.4 104294.43    17
## 57   1.3944216 122355.9  17871.57  140227.4 104484.29    17
## 58   1.2705450 122560.9  17948.03  140508.9 104612.85    17
## 59   1.1576733 122745.8  18030.27  140776.1 104715.54    17
## 60   1.0548288 122924.8  18113.12  141037.9 104811.70    17
## 61   0.9611207 123061.5  18193.03  141254.6 104868.52    17
## 62   0.8757374 123208.1  18284.47  141492.6 104923.62    17
## 63   0.7979393 123382.1  18384.57  141766.7 104997.55    17
## 64   0.7270526 123562.6  18492.35  142055.0 105070.27    17
## 65   0.6624632 123738.4  18590.67  142329.1 105147.78    18
## 66   0.6036118 123872.3  18675.52  142547.8 105196.81    18
## 67   0.5499886 124019.1  18757.67  142776.8 105261.42    18
## 68   0.5011291 124128.8  18833.54  142962.4 105295.28    17
## 69   0.4566102 124241.7  18899.24  143141.0 105342.49    18
## 70   0.4160462 124350.6  18966.30  143316.9 105384.32    18
## 71   0.3790858 124424.4  19016.58  143440.9 105407.79    18
## 72   0.3454089 124475.9  19067.59  143543.5 105408.31    18
## 73   0.3147237 124553.7  19102.89  143656.6 105450.82    18
## 74   0.2867645 124593.8  19149.36  143743.2 105444.46    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   17.19108   83.59338
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
