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
##  [1] 452.5416 450.9198 450.0587 449.7803 449.4874 449.1674 448.8179
##  [8] 448.4362 448.0197 447.5655 447.0703 446.5309 445.9438 445.3051
## [15] 444.6110 443.8572 443.0395 442.1536 441.1948 440.1585 439.0399
## [22] 437.8343 436.5371 435.1439 433.6504 432.0527 430.3473 428.5315
## [29] 426.6031 424.5607 422.4042 420.1345 417.7536 415.2653 412.6746
## [36] 409.9885 407.2154 404.3651 401.4496 398.4820 395.4771 392.4507
## [43] 389.4195 386.4006 383.4116 380.4696 377.5913 374.7923 372.0873
## [50] 369.4880 367.0056 364.6489 362.4248 360.3374 358.3892 356.5804
## [57] 354.9096 353.3738 351.9685 350.6882 349.5267 348.4770 347.5321
## [64] 346.6829 345.9284 345.2581 344.6618 344.1350 343.6699 343.2664
## [71] 342.9171 342.6092 342.3433 342.1138 341.9197 341.7497 341.6031
## [78] 341.4803 341.3693 341.2758 341.1885 341.1127 341.0378 340.9699
## [85] 340.9023 340.8320 340.7599 340.6872 340.6125 340.5284 340.4417
## [92] 340.3497 340.2530 340.1513 340.0447 339.9340 339.8203 339.7038
## [99] 339.5833
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.5833
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.488
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
##  [1] 451.0536 443.4825 433.1665 424.1927 415.5205 406.9885 398.2590
##  [8] 390.4618 383.7895 378.0900 373.2321 368.8841 365.1243 361.7203
## [15] 358.5230 355.5999 352.8199 350.4052 348.3877 346.7188 345.3413
## [22] 344.2101 343.2950 342.5545 341.9495 341.4610 341.0737 340.7979
## [29] 340.6385 340.5838 340.5833 340.6210 340.6616 340.7135 340.8610
## [36] 341.1976 341.7175 342.2338 342.5355 342.2858 341.5490 340.3482
## [43] 339.1785 338.2600 337.4045 336.7890 336.2955 335.9404 335.6874
## [50] 335.5318 335.4851 335.5522 335.7216 335.8253 335.8423 335.8696
## [57] 335.7975 335.6463 335.5500 335.5493 335.6512 335.7524 335.8688
## [64] 336.0120 336.1436 336.2953 336.4186 336.5476 336.6711 336.8173
## [71] 336.9200 337.0510 337.1413 337.2755 337.3638 337.4094
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 335.4851
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 365.1243
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
## 1  255.2820965 203449.4  33806.05  237255.4 169643.31     0
## 2  232.6035386 196676.7  34132.20  230808.9 162544.53     1
## 3  211.9396813 187633.2  32881.08  220514.3 154752.10     2
## 4  193.1115442 179939.4  31698.59  211638.0 148240.85     2
## 5  175.9560468 172657.3  30714.84  203372.1 141942.44     3
## 6  160.3245966 165639.7  29895.27  195534.9 135744.39     4
## 7  146.0818013 158610.2  28870.26  187480.5 129739.96     4
## 8  133.1042967 152460.4  27986.01  180446.4 124474.38     4
## 9  121.2796778 147294.4  27232.00  174526.4 120062.40     4
## 10 110.5055255 142952.0  26598.32  169550.3 116353.70     4
## 11 100.6885192 139302.2  26077.39  165379.6 113224.85     5
## 12  91.7436287 136075.5  25626.68  161702.1 110448.79     5
## 13  83.5933775 133315.7  25255.87  158571.6 108059.85     5
## 14  76.1671723 130841.6  24942.03  155783.6 105899.54     5
## 15  69.4006906 128538.7  24634.16  153172.9 103904.58     6
## 16  63.2353245 126451.3  24334.57  150785.9 102116.74     6
## 17  57.6176726 124481.9  24051.69  148533.6 100430.23     6
## 18  52.4990774 122783.8  23818.74  146602.5  98965.04     6
## 19  47.8352040 121374.0  23626.02  145000.0  97747.98     6
## 20  43.5856563 120213.9  23469.37  143683.3  96744.54     6
## 21  39.7136268 119260.6  23338.31  142598.9  95922.29     6
## 22  36.1855776 118480.6  23233.36  141714.0  95247.24     6
## 23  32.9709506 117851.5  23151.35  141002.8  94700.11     6
## 24  30.0419022 117343.6  23084.34  140427.9  94259.22     6
## 25  27.3730624 116929.5  23027.92  139957.4  93901.56     6
## 26  24.9413150 116595.6  22980.03  139575.6  93615.57     6
## 27  22.7255973 116331.2  22937.61  139268.8  93393.63     6
## 28  20.7067179 116143.2  22896.71  139039.9  93246.48     6
## 29  18.8671902 116034.6  22867.80  138902.4  93166.77     6
## 30  17.1910810 115997.3  22851.22  138848.5  93146.07     7
## 31  15.6638727 115997.0  22839.24  138836.2  93157.72     7
## 32  14.2723374 116022.6  22827.92  138850.6  93194.73     7
## 33  13.0044223 116050.3  22808.43  138858.7  93241.88     9
## 34  11.8491453 116085.7  22786.94  138872.6  93298.75     9
## 35  10.7964999 116186.2  22771.74  138958.0  93414.48     9
## 36   9.8373686 116415.8  22763.86  139179.7  93651.96     9
## 37   8.9634439 116770.9  22758.29  139529.2  94012.58     9
## 38   8.1671562 117124.0  22759.13  139883.1  94364.85    11
## 39   7.4416086 117330.5  22762.94  140093.5  94567.61    11
## 40   6.7805166 117159.6  22616.33  139775.9  94543.25    12
## 41   6.1781542 116655.7  22414.20  139069.9  94241.51    12
## 42   5.6293040 115836.9  22264.17  138101.0  93572.71    13
## 43   5.1292121 115042.1  22151.22  137193.3  92890.84    13
## 44   4.6735471 114419.8  22055.88  136475.7  92363.96    13
## 45   4.2583620 113841.8  21990.18  135832.0  91851.59    13
## 46   3.8800609 113426.9  21958.72  135385.6  91468.15    13
## 47   3.5353670 113094.7  21946.33  135041.0  91148.33    13
## 48   3.2212947 112856.0  21934.83  134790.8  90921.12    13
## 49   2.9351238 112686.0  21926.91  134612.9  90759.11    13
## 50   2.6743755 112581.6  21917.79  134499.4  90663.80    13
## 51   2.4367913 112550.2  21886.22  134436.5  90664.02    13
## 52   2.2203135 112595.3  21837.78  134433.1  90757.53    14
## 53   2.0230670 112709.0  21787.85  134496.8  90921.14    15
## 54   1.8433433 112778.7  21748.65  134527.3  91030.01    15
## 55   1.6795857 112790.0  21715.16  134505.2  91074.88    17
## 56   1.5303760 112808.4  21668.25  134476.6  91140.14    17
## 57   1.3944216 112760.0  21610.01  134370.0  91149.95    17
## 58   1.2705450 112658.5  21565.44  134223.9  91093.01    17
## 59   1.1576733 112593.8  21559.20  134153.0  91034.60    17
## 60   1.0548288 112593.3  21573.76  134167.1  91019.54    17
## 61   0.9611207 112661.7  21582.67  134244.4  91079.05    17
## 62   0.8757374 112729.7  21594.58  134324.3  91135.10    17
## 63   0.7979393 112807.8  21610.48  134418.3  91197.36    17
## 64   0.7270526 112904.1  21619.43  134523.5  91284.65    17
## 65   0.6624632 112992.5  21626.11  134618.6  91366.42    18
## 66   0.6036118 113094.5  21630.90  134725.4  91463.64    18
## 67   0.5499886 113177.4  21634.47  134811.9  91542.97    18
## 68   0.5011291 113264.3  21627.47  134891.7  91636.79    17
## 69   0.4566102 113347.4  21621.40  134968.8  91726.01    18
## 70   0.4160462 113445.9  21615.04  135060.9  91830.86    18
## 71   0.3790858 113515.1  21614.68  135129.8  91900.42    18
## 72   0.3454089 113603.3  21617.70  135221.0  91985.65    18
## 73   0.3147237 113664.2  21626.62  135290.9  92037.62    18
## 74   0.2867645 113754.7  21632.81  135387.6  92121.94    18
## 75   0.2612891 113814.3  21634.02  135448.4  92180.32    18
## 76   0.2380769 113845.1  21643.23  135488.3  92201.89    18
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
