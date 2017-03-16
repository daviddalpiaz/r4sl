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
## (Intercept) 185.946731847
## AtBat         0.096634022
## Hits          0.408580478
## HmRun         1.242303539
## Runs          0.650047295
## RBI           0.642033635
## Walks         0.848737422
## Years         2.608433226
## CAtBat        0.008188531
## CHits         0.031829975
## CHmRun        0.235663247
## CRuns         0.063816873
## CRBI          0.066045116
## CWalks        0.062642350
## LeagueN       4.252099497
## DivisionW   -25.296959330
## PutOuts       0.059902888
## Assists       0.008305300
## Errors       -0.185603402
## NewLeagueN    3.676189338
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
## (Intercept) 185.946731847
## AtBat         0.096634022
## Hits          0.408580478
## HmRun         1.242303539
## Runs          0.650047295
## RBI           0.642033635
## Walks         0.848737422
## Years         2.608433226
## CAtBat        0.008188531
## CHits         0.031829975
## CHmRun        0.235663247
## CRuns         0.063816873
## CRBI          0.066045116
## CWalks        0.062642350
## LeagueN       4.252099497
## DivisionW   -25.296959330
## PutOuts       0.059902888
## Assists       0.008305300
## Errors       -0.185603402
## NewLeagueN    3.676189338
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 681.7166
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 128551
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.7503 450.3993 449.4753 449.2081 448.9160 448.5969 448.2483
##  [8] 447.8678 447.4525 446.9997 446.5062 445.9687 445.3836 444.7474
## [15] 444.0560 443.3054 442.4914 441.6097 440.6557 439.6248 438.5125
## [22] 437.3142 436.0254 434.6417 433.1591 431.5739 429.8829 428.0835
## [29] 426.1737 424.1526 422.0202 419.7777 417.4275 414.9737 412.4216
## [36] 409.7785 407.0530 404.2554 401.3976 398.4931 395.5568 392.6042
## [43] 389.6522 386.7176 383.8177 380.9694 378.1886 375.4907 372.8894
## [50] 370.3963 368.0211 365.7722 363.6558 361.6755 359.8328 358.1276
## [57] 356.5581 355.1207 353.8110 352.6232 351.5512 350.5881 349.7271
## [64] 348.9576 348.2805 347.6871 347.1660 346.7124 346.3168 345.9814
## [71] 345.6999 345.4629 345.2621 345.0989 344.9660 344.8625 344.7794
## [78] 344.7163 344.6665 344.6282 344.5982 344.5730 344.5485 344.5269
## [85] 344.5020 344.4721 344.4366 344.3944 344.3448 344.2868 344.2197
## [92] 344.1436 344.0584 343.9653 343.8645 343.7572 343.6443 343.5273
## [99] 343.4065
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 343.4065
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 365.7722
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
##  [1] 450.5263 441.4187 431.2707 423.1779 416.2570 408.6241 400.7988
##  [8] 393.6132 387.2333 381.9332 377.4826 373.5735 370.0382 366.9317
## [15] 364.0332 361.1574 358.3700 356.0726 354.1607 352.5418 351.2098
## [22] 350.1171 349.2372 348.5172 347.9245 347.4842 347.1981 347.0828
## [29] 347.0488 347.0728 347.1313 347.2110 347.2808 347.3121 347.4146
## [36] 347.9090 348.5962 349.2483 349.4744 349.0223 348.3232 347.4398
## [43] 346.6251 345.7624 344.7665 343.9482 343.2742 342.6811 342.2459
## [50] 342.0123 341.9491 342.1210 342.1432 342.2141 342.3332 342.5035
## [57] 342.6469 342.8082 342.9436 343.0386 343.0436 343.0788 343.1027
## [64] 343.1581 343.1951 343.2734 343.3823 343.4520 343.5516 343.5809
## [71] 343.7100 343.7820 343.8766 343.9779 344.0576
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.9491
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 361.1574
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
## 1  255.2820965 202973.9  25361.00  228334.9 177612.9     0
## 2  232.6035386 194850.4  24941.13  219791.6 169909.3     1
## 3  211.9396813 185994.4  23740.10  209734.5 162254.3     2
## 4  193.1115442 179079.6  22618.25  201697.8 156461.3     2
## 5  175.9560468 173269.9  21527.42  194797.3 151742.4     3
## 6  160.3245966 166973.6  20670.50  187644.1 146303.1     4
## 7  146.0818013 160639.7  19664.90  180304.6 140974.8     4
## 8  133.1042967 154931.4  18710.94  173642.3 136220.4     4
## 9  121.2796778 149949.6  17913.43  167863.0 132036.2     4
## 10 110.5055255 145873.0  17284.77  163157.8 128588.2     4
## 11 100.6885192 142493.1  16811.32  159304.4 125681.8     5
## 12  91.7436287 139557.1  16514.73  156071.9 123042.4     5
## 13  83.5933775 136928.2  16354.70  153283.0 120573.5     5
## 14  76.1671723 134638.9  16229.25  150868.1 118409.6     5
## 15  69.4006906 132520.1  16191.84  148712.0 116328.3     6
## 16  63.2353245 130434.7  16117.32  146552.0 114317.3     6
## 17  57.6176726 128429.0  15980.66  144409.7 112448.4     6
## 18  52.4990774 126787.7  15872.11  142659.8 110915.6     6
## 19  47.8352040 125429.8  15807.18  141236.9 109622.6     6
## 20  43.5856563 124285.7  15783.29  140069.0 108502.4     6
## 21  39.7136268 123348.3  15781.24  139129.5 107567.1     6
## 22  36.1855776 122582.0  15795.16  138377.2 106786.8     6
## 23  32.9709506 121966.6  15817.19  137783.8 106149.5     6
## 24  30.0419022 121464.3  15849.37  137313.6 105614.9     6
## 25  27.3730624 121051.4  15888.00  136939.4 105163.4     6
## 26  24.9413150 120745.2  15929.03  136674.3 104816.2     6
## 27  22.7255973 120546.5  15971.91  136518.5 104574.6     6
## 28  20.7067179 120466.5  16004.81  136471.3 104461.7     6
## 29  18.8671902 120442.9  16039.89  136482.7 104403.0     6
## 30  17.1910810 120459.5  16078.14  136537.6 104381.3     7
## 31  15.6638727 120500.1  16116.15  136616.3 104384.0     7
## 32  14.2723374 120555.5  16159.43  136714.9 104396.0     7
## 33  13.0044223 120604.0  16208.82  136812.8 104395.1     9
## 34  11.8491453 120625.7  16258.99  136884.7 104366.7     9
## 35  10.7964999 120696.9  16311.54  137008.5 104385.4     9
## 36   9.8373686 121040.7  16341.25  137381.9 104699.4     9
## 37   8.9634439 121519.3  16318.84  137838.2 105200.5     9
## 38   8.1671562 121974.4  16261.73  138236.1 105712.7    11
## 39   7.4416086 122132.4  16213.78  138346.2 105918.6    11
## 40   6.7805166 121816.6  16193.77  138010.3 105622.8    12
## 41   6.1781542 121329.1  16058.23  137387.3 105270.8    12
## 42   5.6293040 120714.4  15841.29  136555.7 104873.1    13
## 43   5.1292121 120149.0  15662.36  135811.3 104486.6    13
## 44   4.6735471 119551.6  15508.99  135060.6 104042.6    13
## 45   4.2583620 118864.0  15389.78  134253.7 103474.2    13
## 46   3.8800609 118300.4  15286.76  133587.2 103013.6    13
## 47   3.5353670 117837.2  15203.93  133041.1 102633.3    13
## 48   3.2212947 117430.3  15135.31  132565.6 102295.0    13
## 49   2.9351238 117132.2  15066.31  132198.5 102065.9    13
## 50   2.6743755 116972.4  15019.82  131992.2 101952.6    13
## 51   2.4367913 116929.2  14997.53  131926.7 101931.7    13
## 52   2.2203135 117046.8  15006.93  132053.7 102039.9    14
## 53   2.0230670 117062.0  15004.53  132066.5 102057.5    15
## 54   1.8433433 117110.5  14976.02  132086.5 102134.5    15
## 55   1.6795857 117192.0  14945.13  132137.1 102246.9    17
## 56   1.5303760 117308.7  14924.91  132233.6 102383.8    17
## 57   1.3944216 117406.9  14926.61  132333.5 102480.3    17
## 58   1.2705450 117517.5  14935.01  132452.5 102582.5    17
## 59   1.1576733 117610.3  14952.53  132562.8 102657.8    17
## 60   1.0548288 117675.5  14979.55  132655.0 102695.9    17
## 61   0.9611207 117678.9  15023.74  132702.7 102655.2    17
## 62   0.8757374 117703.1  15061.76  132764.8 102641.3    17
## 63   0.7979393 117719.4  15096.26  132815.7 102623.2    17
## 64   0.7270526 117757.5  15122.66  132880.2 102634.9    17
## 65   0.6624632 117782.9  15144.90  132927.8 102638.0    18
## 66   0.6036118 117836.6  15162.80  132999.4 102673.8    18
## 67   0.5499886 117911.4  15173.41  133084.8 102738.0    18
## 68   0.5011291 117959.3  15186.95  133146.2 102772.3    17
## 69   0.4566102 118027.7  15193.78  133221.5 102833.9    18
## 70   0.4160462 118047.8  15215.54  133263.4 102832.3    18
## 71   0.3790858 118136.6  15224.94  133361.5 102911.6    18
## 72   0.3454089 118186.0  15247.42  133433.5 102938.6    18
## 73   0.3147237 118251.1  15252.27  133503.4 102998.8    18
## 74   0.2867645 118320.8  15269.67  133590.4 103051.1    18
## 75   0.2612891 118375.7  15282.77  133658.4 103092.9    18
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
