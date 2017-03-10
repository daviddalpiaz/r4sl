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
##                        1
## (Intercept)   9.88559247
## AtBat         0.03490091
## Hits          0.99895355
## HmRun         0.16809479
## Runs          1.11069309
## RBI           0.87378443
## Walks         1.79291526
## Years         0.21529403
## CAtBat        0.01116099
## CHits         0.06439189
## CHmRun        0.44865817
## CRuns         0.12805488
## CRBI          0.13636588
## CWalks        0.03088644
## LeagueN      26.87720388
## DivisionW   -90.95870304
## PutOuts       0.19011040
## Assists       0.04162273
## Errors       -1.78525047
## NewLeagueN    7.37938539
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 9060.077
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
##  [1] 451.3850 449.6090 449.0060 448.7339 448.4364 448.1114 447.7564
##  [8] 447.3689 446.9460 446.4848 445.9821 445.4346 444.8387 444.1906
## [15] 443.4863 442.7215 441.8922 440.9939 440.0218 438.9713 437.8377
## [22] 436.6163 435.3026 433.8920 432.3805 430.7642 429.0398 427.2045
## [29] 425.2565 423.1946 421.0188 418.7304 416.3317 413.8269 411.2214
## [36] 408.5227 405.7395 402.8822 399.9631 396.9962 393.9965 390.9804
## [43] 387.9647 384.9673 382.0058 379.0975 376.2593 373.5069 370.8544
## [50] 368.3145 365.8965 363.6098 361.4606 359.4528 357.5881 355.8663
## [57] 354.2856 352.8424 351.5319 350.3483 349.2852 348.3355 347.4920
## [64] 346.7446 346.0912 345.5272 345.0406 344.6234 344.2673 343.9764
## [71] 343.7407 343.5547 343.4109 343.3074 343.2377 343.1989 343.1889
## [78] 343.1976 343.2291 343.2750 343.3323 343.3969 343.4691 343.5431
## [85] 343.6175 343.6905 343.7573 343.8201 343.8742 343.9202 343.9512
## [92] 343.9792 343.9835 343.9962 343.9681 343.9606 343.9063 343.8807
## [99] 343.8044
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 343.1889
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 368.3145
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
##  [1] 449.7035 443.0867 433.4144 425.2424 417.7431 409.8680 401.4207
##  [8] 393.9455 387.5565 382.1095 377.4535 373.3291 369.5605 366.0061
## [15] 362.8988 360.1750 357.2635 354.6401 352.4273 350.5810 349.0367
## [22] 347.7655 346.7678 346.0552 345.4689 345.0063 344.6812 344.5128
## [29] 344.4155 344.3833 344.3883 344.3958 344.3527 344.3649 344.5295
## [36] 345.0496 345.8248 346.5018 346.3733 345.7615 345.0199 344.2749
## [43] 343.6254 343.0023 342.2685 341.6507 341.1887 340.8170 340.5560
## [50] 340.4395 340.4039 340.5006 340.6245 340.7845 341.0475 341.3319
## [57] 341.6937 342.1048 342.5081 342.8772 343.2239 343.5402 343.8207
## [64] 344.0828 344.3381 344.5793 344.8036 345.0680 345.3256 345.5804
## [71] 345.7838 346.0056 346.1855
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.4039
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 366.0061
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
## 1  255.2820965 202233.3  24458.62  226691.9 177774.66     0
## 2  232.6035386 196325.8  24999.11  221324.9 171326.71     1
## 3  211.9396813 187848.0  23760.73  211608.8 164087.30     2
## 4  193.1115442 180831.1  22669.46  203500.6 158161.63     2
## 5  175.9560468 174509.3  21601.24  196110.5 152908.04     3
## 6  160.3245966 167991.8  20590.47  188582.3 147401.34     4
## 7  146.0818013 161138.5  19855.84  180994.4 141282.70     4
## 8  133.1042967 155193.0  19256.68  174449.7 135936.34     4
## 9  121.2796778 150200.1  18851.33  169051.4 131348.73     4
## 10 110.5055255 146007.7  18612.76  164620.4 127394.91     4
## 11 100.6885192 142471.2  18499.88  160971.1 123971.30     5
## 12  91.7436287 139374.6  18478.14  157852.8 120896.50     5
## 13  83.5933775 136575.0  18556.85  155131.8 118018.10     5
## 14  76.1671723 133960.5  18520.26  152480.7 115440.20     5
## 15  69.4006906 131695.5  18544.53  150240.0 113150.98     6
## 16  63.2353245 129726.1  18616.91  148343.0 111109.14     6
## 17  57.6176726 127637.2  18582.23  146219.5 109054.99     6
## 18  52.4990774 125769.6  18566.40  144336.0 107203.23     6
## 19  47.8352040 124205.0  18573.28  142778.3 105631.72     6
## 20  43.5856563 122907.1  18597.05  141504.1 104310.02     6
## 21  39.7136268 121826.6  18631.17  140457.8 103195.45     6
## 22  36.1855776 120940.8  18671.85  139612.7 102268.97     6
## 23  32.9709506 120247.9  18718.41  138966.3 101529.53     6
## 24  30.0419022 119754.2  18770.10  138524.3 100984.11     6
## 25  27.3730624 119348.8  18821.71  138170.5 100527.08     6
## 26  24.9413150 119029.4  18871.57  137900.9 100157.79     6
## 27  22.7255973 118805.1  18915.91  137721.0  99889.22     6
## 28  20.7067179 118689.0  18952.04  137641.1  99737.01     6
## 29  18.8671902 118622.0  18985.86  137607.9  99636.18     6
## 30  17.1910810 118599.9  19014.63  137614.5  99585.22     7
## 31  15.6638727 118603.3  19042.36  137645.7  99560.95     7
## 32  14.2723374 118608.5  19069.69  137678.2  99538.79     7
## 33  13.0044223 118578.8  19087.12  137665.9  99491.66     9
## 34  11.8491453 118587.2  19087.25  137674.4  99499.92     9
## 35  10.7964999 118700.6  19091.22  137791.8  99609.37     9
## 36   9.8373686 119059.2  19141.99  138201.2  99917.23     9
## 37   8.9634439 119594.8  19202.83  138797.6 100391.93     9
## 38   8.1671562 120063.5  19238.96  139302.4 100824.51    11
## 39   7.4416086 119974.4  19255.53  139230.0 100718.92    11
## 40   6.7805166 119551.0  19303.31  138854.4 100247.73    12
## 41   6.1781542 119038.7  19343.25  138381.9  99695.45    12
## 42   5.6293040 118525.2  19382.09  137907.3  99143.08    13
## 43   5.1292121 118078.4  19411.89  137490.3  98666.51    13
## 44   4.6735471 117650.6  19408.57  137059.1  98242.00    13
## 45   4.2583620 117147.7  19386.12  136533.9  97761.62    13
## 46   3.8800609 116725.2  19368.27  136093.5  97356.91    13
## 47   3.5353670 116409.8  19354.25  135764.0  97055.51    13
## 48   3.2212947 116156.3  19342.98  135499.2  96813.28    13
## 49   2.9351238 115978.4  19329.44  135307.9  96648.98    13
## 50   2.6743755 115899.1  19318.10  135217.2  96580.99    13
## 51   2.4367913 115874.8  19309.68  135184.5  96565.13    13
## 52   2.2203135 115940.7  19293.76  135234.4  96646.91    14
## 53   2.0230670 116025.0  19259.29  135284.3  96765.74    15
## 54   1.8433433 116134.1  19204.53  135338.6  96929.55    15
## 55   1.6795857 116313.4  19152.90  135466.3  97160.53    17
## 56   1.5303760 116507.5  19102.03  135609.5  97405.43    17
## 57   1.3944216 116754.6  19038.30  135792.9  97716.31    17
## 58   1.2705450 117035.7  18982.67  136018.4  98053.01    17
## 59   1.1576733 117311.8  18935.79  136247.6  98376.03    17
## 60   1.0548288 117564.8  18898.57  136463.3  98666.19    17
## 61   0.9611207 117802.6  18866.80  136669.4  98935.85    17
## 62   0.8757374 118019.9  18840.24  136860.1  99179.64    17
## 63   0.7979393 118212.7  18815.14  137027.8  99397.56    17
## 64   0.7270526 118393.0  18791.73  137184.7  99601.23    17
## 65   0.6624632 118568.7  18771.27  137340.0  99797.47    18
## 66   0.6036118 118734.9  18753.95  137488.8  99980.91    18
## 67   0.5499886 118889.5  18737.11  137626.6 100152.39    18
## 68   0.5011291 119071.9  18716.15  137788.1 100355.78    17
## 69   0.4566102 119249.8  18707.89  137957.7 100541.88    18
## 70   0.4160462 119425.8  18681.46  138107.3 100744.38    18
## 71   0.3790858 119566.5  18673.08  138239.5 100893.38    18
## 72   0.3454089 119719.9  18667.18  138387.0 101052.68    18
## 73   0.3147237 119844.4  18655.11  138499.5 101189.32    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   76.16717
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
