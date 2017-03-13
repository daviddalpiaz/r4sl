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
##                         1
## (Intercept)  1.226775e+01
## AtBat       -2.419845e-02
## Hits         1.160708e+00
## HmRun       -2.631285e-01
## Runs         1.143697e+00
## RBI          8.628801e-01
## Walks        1.966769e+00
## Years       -1.136139e+00
## CAtBat       1.065732e-02
## CHits        7.226563e-02
## CHmRun       4.922058e-01
## CRuns        1.434083e-01
## CRBI         1.528435e-01
## CWalks       9.709707e-04
## LeagueN      3.150933e+01
## DivisionW   -1.003989e+02
## PutOuts      2.096675e-01
## Assists      5.665892e-02
## Errors      -2.194190e+00
## NewLeagueN   4.428640e+00
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 11106.17
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
##  [1] 453.3518 451.6035 450.9435 450.6759 450.3834 450.0637 449.7145
##  [8] 449.3333 448.9173 448.4635 447.9690 447.4302 446.8438 446.2060
## [15] 445.5127 444.7599 443.9435 443.0589 442.1015 441.0668 439.9500
## [22] 438.7464 437.4515 436.0608 434.5700 432.9754 431.2735 429.4614
## [29] 427.5371 425.4992 423.3476 421.0830 418.7077 416.2252 413.6407
## [36] 410.9608 408.1940 405.3501 402.4406 399.4788 396.4790 393.4569
## [43] 390.4288 387.4118 384.4228 381.4790 378.5965 375.7909 373.0761
## [50] 370.4647 367.9671 365.5914 363.3452 361.2327 359.2563 357.4166
## [57] 355.7124 354.1411 352.6987 351.3803 350.1800 349.0918 348.1092
## [64] 347.2246 346.4344 345.7327 345.1092 344.5593 344.0762 343.6580
## [71] 343.3000 342.9935 342.7342 342.5232 342.3504 342.2137 342.1077
## [78] 342.0317 341.9828 341.9550 341.9476 341.9553 341.9773 342.0117
## [85] 342.0539 342.1035 342.1604 342.2199 342.2808 342.3439 342.4063
## [92] 342.4698 342.5303 342.5896 342.6472 342.7029 342.7569 342.8076
## [99] 342.8588
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.9476
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.4647
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
## (Intercept) 127.95694754
## AtBat         .         
## Hits          1.42342566
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.58214111
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.16027975
## CRBI          0.33667715
## CWalks        .         
## LeagueN       .         
## DivisionW    -8.06171262
## PutOuts       0.08393604
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
## (Intercept) 127.95694754
## AtBat         .         
## Hits          1.42342566
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.58214111
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.16027975
## CRBI          0.33667715
## CWalks        .         
## LeagueN       .         
## DivisionW    -8.06171262
## PutOuts       0.08393604
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 11.64817
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 118581.5
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 452.5798 442.6270 432.5479 423.9082 415.7373 406.9694 398.4639
##  [8] 391.0784 385.1085 380.1094 375.8270 371.9309 368.4495 365.3922
## [15] 362.4783 359.6434 356.9659 354.5782 352.5041 350.7784 349.3445
## [22] 348.1644 347.1965 346.3964 345.7363 345.1833 344.6989 344.3614
## [29] 344.1581 344.0415 343.9570 343.9006 343.8389 343.7512 343.6340
## [36] 343.6358 343.9060 344.1955 344.1307 343.9638 343.8581 343.6837
## [43] 343.3056 343.0539 342.5859 341.9393 341.4370 341.0587 340.7964
## [50] 340.5906 340.4933 340.6047 340.7740 341.0149 341.3589 341.7488
## [57] 342.1171 342.4092 342.6318 342.7387 342.9363 343.1180 343.3604
## [64] 343.5171 343.6918 343.8314 344.0014 344.1231 344.2732 344.3512
## [71] 344.4451 344.5120 344.6181 344.6526
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.4933
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 362.4783
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
## 1  255.2820965 204828.5  26758.59  231587.1 178069.90     0
## 2  232.6035386 195918.6  26754.04  222672.7 169164.61     1
## 3  211.9396813 187097.7  26297.30  213395.0 160800.39     2
## 4  193.1115442 179698.1  25894.20  205592.3 153803.94     2
## 5  175.9560468 172837.5  25515.98  198353.5 147321.51     3
## 6  160.3245966 165624.1  25101.65  190725.7 140522.40     4
## 7  146.0818013 158773.5  24761.49  183535.0 134012.01     4
## 8  133.1042967 152942.3  24470.39  177412.7 128471.90     4
## 9  121.2796778 148308.6  24167.19  172475.8 124141.40     4
## 10 110.5055255 144483.1  23902.70  168385.8 120580.44     4
## 11 100.6885192 141246.0  23643.39  164889.4 117602.57     5
## 12  91.7436287 138332.6  23383.11  161715.7 114949.48     5
## 13  83.5933775 135755.0  23132.71  158887.7 112622.31     5
## 14  76.1671723 133511.5  22841.72  156353.2 110669.76     5
## 15  69.4006906 131390.6  22563.17  153953.7 108827.38     6
## 16  63.2353245 129343.4  22284.92  151628.3 107058.43     6
## 17  57.6176726 127424.7  21992.03  149416.7 105432.66     6
## 18  52.4990774 125725.7  21658.87  147384.5 104066.81     6
## 19  47.8352040 124259.2  21315.07  145574.2 102944.10     6
## 20  43.5856563 123045.5  21013.80  144059.3 102031.71     6
## 21  39.7136268 122041.6  20750.13  142791.7 101291.43     6
## 22  36.1855776 121218.5  20520.90  141739.4 100697.58     6
## 23  32.9709506 120545.4  20322.14  140867.6 100223.30     6
## 24  30.0419022 119990.5  20147.16  140137.6  99843.30     6
## 25  27.3730624 119533.6  19993.32  139526.9  99540.28     6
## 26  24.9413150 119151.5  19857.25  139008.8  99294.26     6
## 27  22.7255973 118817.3  19736.78  138554.1  99080.52     6
## 28  20.7067179 118584.7  19626.81  138211.6  98957.94     6
## 29  18.8671902 118444.8  19526.97  137971.8  98917.84     6
## 30  17.1910810 118364.5  19428.23  137792.8  98936.32     7
## 31  15.6638727 118306.4  19336.17  137642.6  98970.23     7
## 32  14.2723374 118267.6  19246.74  137514.4  99020.88     7
## 33  13.0044223 118225.2  19158.92  137384.1  99066.30     9
## 34  11.8491453 118164.9  19071.97  137236.8  99092.90     9
## 35  10.7964999 118084.3  18987.85  137072.1  99096.46     9
## 36   9.8373686 118085.5  18904.31  136989.9  99181.23     9
## 37   8.9634439 118271.3  18803.09  137074.4  99468.24     9
## 38   8.1671562 118470.6  18714.66  137185.2  99755.91    11
## 39   7.4416086 118426.0  18653.42  137079.4  99772.53    11
## 40   6.7805166 118311.1  18586.41  136897.5  99724.71    12
## 41   6.1781542 118238.4  18502.41  136740.8  99735.96    12
## 42   5.6293040 118118.5  18297.60  136416.1  99820.89    13
## 43   5.1292121 117858.7  17922.18  135780.9  99936.55    13
## 44   4.6735471 117686.0  17637.71  135323.7 100048.30    13
## 45   4.2583620 117365.1  17390.12  134755.2  99974.96    13
## 46   3.8800609 116922.5  17157.40  134079.9  99765.12    13
## 47   3.5353670 116579.2  16963.03  133542.2  99616.17    13
## 48   3.2212947 116321.0  16789.05  133110.1  99531.99    13
## 49   2.9351238 116142.2  16631.79  132774.0  99510.43    13
## 50   2.6743755 116001.9  16488.58  132490.5  99513.34    13
## 51   2.4367913 115935.7  16374.74  132310.4  99560.96    13
## 52   2.2203135 116011.6  16273.55  132285.1  99738.04    14
## 53   2.0230670 116126.9  16186.21  132313.2  99940.73    15
## 54   1.8433433 116291.1  16113.71  132404.9 100177.43    15
## 55   1.6795857 116525.9  16051.07  132577.0 100474.86    17
## 56   1.5303760 116792.2  15991.41  132783.7 100800.82    17
## 57   1.3944216 117044.1  15921.80  132965.9 101122.28    17
## 58   1.2705450 117244.1  15861.95  133106.0 101382.15    17
## 59   1.1576733 117396.6  15783.24  133179.8 101613.34    17
## 60   1.0548288 117469.8  15680.16  133150.0 101789.63    17
## 61   0.9611207 117605.3  15607.23  133212.5 101998.05    17
## 62   0.8757374 117729.9  15543.83  133273.8 102186.10    17
## 63   0.7979393 117896.4  15492.35  133388.8 102404.04    17
## 64   0.7270526 118004.0  15447.84  133451.9 102556.18    17
## 65   0.6624632 118124.0  15409.76  133533.8 102714.26    18
## 66   0.6036118 118220.0  15377.56  133597.6 102842.45    18
## 67   0.5499886 118336.9  15349.66  133686.6 102987.29    18
## 68   0.5011291 118420.7  15329.00  133749.7 103091.72    17
## 69   0.4566102 118524.0  15310.57  133834.6 103213.44    18
## 70   0.4160462 118577.8  15286.35  133864.1 103291.43    18
## 71   0.3790858 118642.4  15272.20  133914.6 103370.22    18
## 72   0.3454089 118688.5  15254.51  133943.0 103434.02    18
## 73   0.3147237 118761.7  15253.51  134015.2 103508.15    18
## 74   0.2867645 118785.4  15240.04  134025.4 103545.36    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   69.40069
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
