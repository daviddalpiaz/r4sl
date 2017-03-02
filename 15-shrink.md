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
## (Intercept) 254.518230141
## AtBat         0.079408204
## Hits          0.317375220
## HmRun         1.065097243
## Runs          0.515835607
## RBI           0.520504723
## Walks         0.661891621
## Years         2.231379426
## CAtBat        0.006679258
## CHits         0.025455999
## CHmRun        0.189661478
## CRuns         0.051057906
## CRBI          0.052776153
## CWalks        0.052170266
## LeagueN       2.114989228
## DivisionW   -17.479743519
## PutOuts       0.043039515
## Assists       0.006296277
## Errors       -0.099487300
## NewLeagueN    2.085946064
```

```r
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  10.275516216
## AtBat         0.008088527
## Hits          1.075046175
## HmRun        -0.046053688
## Runs          1.128360908
## RBI           0.868722584
## Walks         1.876642475
## Years        -0.425145113
## CAtBat        0.010952415
## CHits         0.068113575
## CHmRun        0.469845085
## CRuns         0.135302716
## CRBI          0.144283368
## CWalks        0.017175670
## LeagueN      29.161665212
## DivisionW   -95.881063089
## PutOuts       0.200221069
## Assists       0.048671621
## Errors       -1.988817468
## NewLeagueN    6.058926651
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 10091.44
```

```r
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 254.518230141
## AtBat         0.079408204
## Hits          0.317375220
## HmRun         1.065097243
## Runs          0.515835607
## RBI           0.520504723
## Walks         0.661891621
## Years         2.231379426
## CAtBat        0.006679258
## CHits         0.025455999
## CHmRun        0.189661478
## CRuns         0.051057906
## CRBI          0.052776153
## CWalks        0.052170266
## LeagueN       2.114989228
## DivisionW   -17.479743519
## PutOuts       0.043039515
## Assists       0.006296277
## Errors       -0.099487300
## NewLeagueN    2.085946064
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 321.618
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 138732.7
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 452.5432 450.8526 450.1119 449.8428 449.5486 449.2272 448.8761
##  [8] 448.4928 448.0745 447.6183 447.1211 446.5795 445.9900 445.3488
## [15] 444.6520 443.8953 443.0747 442.1856 441.2236 440.1838 439.0617
## [22] 437.8526 436.5518 435.1550 433.6580 432.0568 430.3483 428.5295
## [29] 426.5985 424.5541 422.3962 420.1257 417.7450 415.2578 412.6695
## [36] 409.9872 407.2193 404.3758 401.4688 398.5118 395.5193 392.5072
## [43] 389.4923 386.4917 383.5228 380.6026 377.7476 374.9733 372.2939
## [50] 369.7214 367.2657 364.9357 362.7383 360.6773 358.7547 356.9708
## [57] 355.3241 353.8115 352.4287 351.1704 350.0306 349.0029 348.0804
## [64] 347.2549 346.5233 345.8764 345.3111 344.8191 344.3897 344.0250
## [71] 343.7179 343.4597 343.2496 343.0843 342.9529 342.8581 342.7933
## [78] 342.7521 342.7325 342.7327 342.7482 342.7748 342.8101 342.8506
## [85] 342.8937 342.9420 342.9855 343.0255 343.0632 343.0936 343.1175
## [92] 343.1331 343.1391 343.1366 343.1243 343.1029 343.0712 343.0308
## [99] 342.9845
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.7325
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 377.7476
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
## (Intercept)  110.6899329
## AtBat         -1.3941293
## Hits           5.3222676
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.4559020
## Years         -8.7478268
## CAtBat         .        
## CHits          .        
## CHmRun         0.4476611
## CRuns          0.6091444
## CRBI           0.3987243
## CWalks        -0.4665820
## LeagueN       31.1402195
## DivisionW   -119.1992215
## PutOuts        0.2682614
## Assists        0.1444617
## Errors        -1.8154257
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 174.4098
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
##  [1] 448.7629 440.5548 430.3626 421.8468 413.8058 405.1364 397.1974
##  [8] 390.2007 384.2565 379.2871 375.1138 371.4390 368.1967 365.1753
## [15] 362.4482 360.0062 357.5925 355.4741 353.7728 352.4260 351.3943
## [22] 350.5985 349.9263 349.3563 348.8837 348.5263 348.3111 348.2887
## [29] 348.3962 348.5474 348.6896 348.8743 349.2180 349.5911 349.9491
## [36] 350.4187 351.2237 352.1534 352.6485 352.7350 352.1902 351.3905
## [43] 350.4844 349.7188 349.1689 348.7496 348.4488 348.2740 348.4300
## [50] 348.5914 348.8446 349.3604 350.0474 350.7402 351.3122 351.8123
## [57] 352.2542 352.6613 352.9973 353.2828 353.4809 353.7938 354.0891
## [64] 354.3580 354.5909 354.8676 355.0688 355.2946 355.4797 355.6100
## [71] 355.7611
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 348.274
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.439
```

## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
#lassoModCV
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high conf.low nzero
## 1  255.2820965 201388.2  22947.17  224335.3 178441.0     0
## 2  232.6035386 194088.5  23252.69  217341.2 170835.9     1
## 3  211.9396813 185212.0  22492.01  207704.0 162719.9     2
## 4  193.1115442 177954.7  21836.97  199791.7 156117.7     2
## 5  175.9560468 171235.3  21175.37  192410.6 150059.9     3
## 6  160.3245966 164135.5  20537.30  184672.8 143598.2     4
## 7  146.0818013 157765.8  20070.75  177836.5 137695.0     4
## 8  133.1042967 152256.6  19696.55  171953.1 132560.0     4
## 9  121.2796778 147653.1  19388.10  167041.2 128265.0     4
## 10 110.5055255 143858.7  19128.53  162987.3 124730.2     4
## 11 100.6885192 140710.4  18963.04  159673.4 121747.4     5
## 12  91.7436287 137966.9  18876.18  156843.1 119090.7     5
## 13  83.5933775 135568.8  18744.62  154313.4 116824.2     5
## 14  76.1671723 133353.0  18531.00  151884.0 114822.0     5
## 15  69.4006906 131368.7  18382.83  149751.5 112985.8     6
## 16  63.2353245 129604.5  18278.33  147882.8 111326.1     6
## 17  57.6176726 127872.4  18122.51  145994.9 109749.9     6
## 18  52.4990774 126361.9  17990.08  144351.9 108371.8     6
## 19  47.8352040 125155.2  17913.62  143068.8 107241.6     6
## 20  43.5856563 124204.1  17879.89  142084.0 106324.2     6
## 21  39.7136268 123477.9  17882.07  141360.0 105595.9     6
## 22  36.1855776 122919.3  17905.64  140824.9 105013.7     6
## 23  32.9709506 122448.4  17939.73  140388.1 104508.7     6
## 24  30.0419022 122049.8  17980.43  140030.3 104069.4     6
## 25  27.3730624 121719.9  18028.15  139748.0 103691.7     6
## 26  24.9413150 121470.6  18080.07  139550.7 103390.5     6
## 27  22.7255973 121320.6  18131.84  139452.5 103188.8     6
## 28  20.7067179 121305.0  18181.47  139486.5 103123.6     6
## 29  18.8671902 121379.9  18225.24  139605.2 103154.7     6
## 30  17.1910810 121485.3  18268.34  139753.6 103216.9     7
## 31  15.6638727 121584.5  18305.85  139890.3 103278.6     7
## 32  14.2723374 121713.3  18344.48  140057.7 103368.8     7
## 33  13.0044223 121953.2  18412.72  140366.0 103540.5     9
## 34  11.8491453 122213.9  18490.80  140704.7 103723.1     9
## 35  10.7964999 122464.4  18568.14  141032.5 103896.2     9
## 36   9.8373686 122793.3  18642.06  141435.3 104151.2     9
## 37   8.9634439 123358.1  18691.74  142049.9 104666.4     9
## 38   8.1671562 124012.0  18779.95  142792.0 105232.1    11
## 39   7.4416086 124360.9  18857.37  143218.3 105503.6    11
## 40   6.7805166 124422.0  18901.77  143323.8 105520.2    12
## 41   6.1781542 124037.9  18837.89  142875.8 105200.0    12
## 42   5.6293040 123475.3  18651.35  142126.6 104823.9    13
## 43   5.1292121 122839.3  18443.57  141282.9 104395.8    13
## 44   4.6735471 122303.3  18271.30  140574.6 104032.0    13
## 45   4.2583620 121918.9  18142.06  140061.0 103776.9    13
## 46   3.8800609 121626.3  18028.97  139655.2 103597.3    13
## 47   3.5353670 121416.5  17920.43  139337.0 103496.1    13
## 48   3.2212947 121294.8  17811.17  139105.9 103483.6    13
## 49   2.9351238 121403.4  17701.11  139104.5 103702.3    13
## 50   2.6743755 121516.0  17598.77  139114.8 103917.2    13
## 51   2.4367913 121692.6  17518.97  139211.6 104173.6    13
## 52   2.2203135 122052.7  17483.77  139536.5 104568.9    14
## 53   2.0230670 122533.2  17475.84  140009.0 105057.3    15
## 54   1.8433433 123018.7  17493.94  140512.6 105524.7    15
## 55   1.6795857 123420.2  17494.77  140915.0 105925.5    17
## 56   1.5303760 123771.9  17471.65  141243.6 106300.2    17
## 57   1.3944216 124083.0  17442.18  141525.2 106640.8    17
## 58   1.2705450 124370.0  17442.94  141812.9 106927.0    17
## 59   1.1576733 124607.1  17465.79  142072.9 107141.3    17
## 60   1.0548288 124808.7  17498.24  142307.0 107310.5    17
## 61   0.9611207 124948.7  17523.37  142472.1 107425.4    17
## 62   0.8757374 125170.0  17566.95  142737.0 107603.1    17
## 63   0.7979393 125379.1  17613.55  142992.6 107765.6    17
## 64   0.7270526 125569.6  17666.83  143236.4 107902.8    17
## 65   0.6624632 125734.7  17714.07  143448.8 108020.6    18
## 66   0.6036118 125931.0  17771.49  143702.5 108159.5    18
## 67   0.5499886 126073.8  17813.53  143887.4 108260.3    18
## 68   0.5011291 126234.2  17857.83  144092.1 108376.4    17
## 69   0.4566102 126365.8  17897.70  144263.5 108468.1    18
## 70   0.4160462 126458.5  17921.58  144380.1 108536.9    18
## 71   0.3790858 126566.0  17960.62  144526.6 108605.4    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   3.221295   91.74363
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
