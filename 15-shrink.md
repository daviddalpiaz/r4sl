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
##  [1] 451.5005 449.9343 449.0735 448.8065 448.5147 448.1958 447.8475
##  [8] 447.4672 447.0523 446.5998 446.1067 445.5695 444.9849 444.3491
## [15] 443.6582 442.9080 442.0945 441.2133 440.2599 439.2296 438.1179
## [22] 436.9201 435.6319 434.2488 432.7668 431.1822 429.4918 427.6928
## [29] 425.7835 423.7629 421.6308 419.3886 417.0387 414.5851 412.0332
## [36] 409.3902 406.6650 403.8677 401.0102 398.1062 395.1705 392.2190
## [43] 389.2683 386.3356 383.4381 380.5928 377.8160 375.1227 372.5267
## [50] 370.0401 367.6724 365.4320 363.3250 361.3549 359.5233 357.8299
## [57] 356.2725 354.8477 353.5507 352.3756 351.3160 350.3651 349.5139
## [64] 348.7571 348.0939 347.5105 346.9992 346.5524 346.1684 345.8466
## [71] 345.5784 345.3503 345.1657 345.0214 344.9058 344.8217 344.7620
## [78] 344.7255 344.7090 344.7058 344.7109 344.7290 344.7501 344.7768
## [85] 344.8039 344.8309 344.8523 344.8728 344.8842 344.8876 344.8833
## [92] 344.8710 344.8479 344.8142 344.7709 344.7180 344.6559 344.5848
## [99] 344.5076
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.5076
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 380.5928
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
##                         1
## (Intercept)  134.48030406
## AtBat         -1.67572220
## Hits           5.94122316
## HmRun          0.04746835
## Runs           .         
## RBI            .         
## Walks          4.95676182
## Years        -10.26657309
## CAtBat         .         
## CHits          .         
## CHmRun         0.56236426
## CRuns          0.70135135
## CRBI           0.38727139
## CWalks        -0.58111548
## LeagueN       32.92255640
## DivisionW   -119.37941356
## PutOuts        0.27580087
## Assists        0.19782326
## Errors        -2.26242857
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 180.1579
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
##  [1] 449.9504 440.2387 430.1610 421.8516 414.4253 406.4396 397.9328
##  [8] 390.6482 384.2091 378.8007 374.4027 370.7548 367.4870 364.5971
## [15] 361.8859 359.1072 356.5840 354.3595 352.5122 350.8613 349.4797
## [22] 348.3240 347.4402 346.8062 346.3247 346.0141 345.8535 345.7518
## [29] 345.7326 345.7453 345.7504 345.7473 345.7601 345.8049 345.8658
## [36] 345.9178 346.0167 346.0952 345.8508 345.2862 344.4724 343.5667
## [43] 342.4727 341.3590 340.4546 339.7584 339.2035 338.7535 338.3421
## [50] 338.0701 337.9248 337.8983 337.9460 338.0873 338.2459 338.4688
## [57] 338.6626 338.7816 338.7568 338.7064 338.7149 338.7688 338.8137
## [64] 338.8976 338.9799 339.0603 339.1521 339.2159 339.3263 339.4017
## [71] 339.4992
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.8983
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.487
```

## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
#lassoModCV
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high  conf.low nzero
## 1  255.2820965 202455.3  21924.88  224380.2 180530.44     0
## 2  232.6035386 193810.1  21601.66  215411.8 172208.47     1
## 3  211.9396813 185038.5  20736.73  205775.2 164301.77     2
## 4  193.1115442 177958.8  20090.60  198049.4 157868.20     2
## 5  175.9560468 171748.4  19576.80  191325.2 152171.56     3
## 6  160.3245966 165193.1  19137.65  184330.8 146055.46     4
## 7  146.0818013 158350.5  18829.60  177180.1 139520.90     4
## 8  133.1042967 152606.0  18693.66  171299.7 133912.39     4
## 9  121.2796778 147616.6  18650.42  166267.1 128966.21     4
## 10 110.5055255 143490.0  18715.96  162206.0 124774.05     4
## 11 100.6885192 140177.4  18864.98  159042.3 121312.38     5
## 12  91.7436287 137459.1  19107.30  156566.4 118351.82     5
## 13  83.5933775 135046.7  19307.77  154354.5 115738.92     5
## 14  76.1671723 132931.1  19569.77  152500.8 113361.28     5
## 15  69.4006906 130961.4  19885.19  150846.6 111076.24     6
## 16  63.2353245 128958.0  20182.86  149140.9 108775.16     6
## 17  57.6176726 127152.2  20430.01  147582.2 106722.18     6
## 18  52.4990774 125570.6  20645.19  146215.8 104925.44     6
## 19  47.8352040 124264.8  20860.71  145125.5 103404.12     6
## 20  43.5856563 123103.6  21036.31  144139.9 102067.33     6
## 21  39.7136268 122136.1  21214.60  143350.7 100921.48     6
## 22  36.1855776 121329.6  21395.06  142724.7  99934.55     6
## 23  32.9709506 120714.7  21575.94  142290.6  99138.75     6
## 24  30.0419022 120274.5  21755.20  142029.7  98519.31     6
## 25  27.3730624 119940.8  21926.15  141866.9  98014.64     6
## 26  24.9413150 119725.8  22083.68  141809.4  97642.07     6
## 27  22.7255973 119614.7  22241.06  141855.7  97373.61     6
## 28  20.7067179 119544.3  22385.17  141929.5  97159.12     6
## 29  18.8671902 119531.0  22517.81  142048.8  97013.21     6
## 30  17.1910810 119539.8  22641.55  142181.4  96898.26     7
## 31  15.6638727 119543.3  22755.98  142299.3  96787.35     7
## 32  14.2723374 119541.2  22866.99  142408.2  96674.20     7
## 33  13.0044223 119550.0  22973.52  142523.6  96576.52     9
## 34  11.8491453 119581.1  23072.92  142654.0  96508.14     9
## 35  10.7964999 119623.2  23165.98  142789.2  96457.19     9
## 36   9.8373686 119659.1  23255.29  142914.4  96403.84     9
## 37   8.9634439 119727.6  23335.00  143062.6  96392.59     9
## 38   8.1671562 119781.9  23524.93  143306.8  96256.93    11
## 39   7.4416086 119612.8  23696.20  143309.0  95916.60    11
## 40   6.7805166 119222.5  23843.17  143065.7  95379.36    12
## 41   6.1781542 118661.2  23799.19  142460.4  94862.02    12
## 42   5.6293040 118038.1  23676.34  141714.4  94361.74    13
## 43   5.1292121 117287.5  23527.47  140815.0  93760.05    13
## 44   4.6735471 116525.9  23405.74  139931.7  93120.20    13
## 45   4.2583620 115909.4  23297.00  139206.4  92612.37    13
## 46   3.8800609 115435.8  23167.76  138603.6  92268.03    13
## 47   3.5353670 115059.0  23056.59  138115.6  92002.41    13
## 48   3.2212947 114753.9  22933.28  137687.2  91820.64    13
## 49   2.9351238 114475.4  22765.91  137241.3  91709.49    13
## 50   2.6743755 114291.4  22636.05  136927.4  91655.32    13
## 51   2.4367913 114193.2  22509.47  136702.6  91683.69    13
## 52   2.2203135 114175.2  22403.99  136579.2  91771.24    14
## 53   2.0230670 114207.5  22310.29  136517.8  91897.24    15
## 54   1.8433433 114303.0  22224.40  136527.4  92078.60    15
## 55   1.6795857 114410.3  22143.72  136554.0  92266.55    17
## 56   1.5303760 114561.2  22069.58  136630.7  92491.57    17
## 57   1.3944216 114692.3  22001.33  136693.7  92691.01    17
## 58   1.2705450 114773.0  21925.15  136698.1  92847.82    17
## 59   1.1576733 114756.2  21824.86  136581.0  92931.32    17
## 60   1.0548288 114722.0  21728.45  136450.5  92993.57    17
## 61   0.9611207 114727.8  21649.07  136376.9  93078.72    17
## 62   0.8757374 114764.3  21583.51  136347.8  93180.79    17
## 63   0.7979393 114794.7  21524.76  136319.5  93269.93    17
## 64   0.7270526 114851.6  21478.04  136329.6  93373.52    17
## 65   0.6624632 114907.4  21439.80  136347.2  93467.57    18
## 66   0.6036118 114961.9  21420.55  136382.4  93541.33    18
## 67   0.5499886 115024.1  21398.07  136422.2  93626.07    18
## 68   0.5011291 115067.4  21378.36  136445.8  93689.08    17
## 69   0.4566102 115142.4  21358.50  136500.9  93783.86    18
## 70   0.4160462 115193.5  21335.83  136529.3  93857.68    18
## 71   0.3790858 115259.7  21321.90  136581.6  93937.80    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.220313   83.59338
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

where $L$ is the appropriate negative log-likelihood.


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

We can extracting the two relevant $\lambda$ values.


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
- [`glmnet` with `caret`](https://github.com/topepo/caret/issues/116) - Some details on Elastic Net tuning in the `caret` package. TODO: move this to elastic net chapter.


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
