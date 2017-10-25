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
## (Intercept) 240.682852262
## AtBat         0.083042199
## Hits          0.334990595
## HmRun         1.105720594
## Runs          0.542496738
## RBI           0.545363022
## Walks         0.698162048
## Years         2.316374820
## CAtBat        0.006988341
## CHits         0.026721778
## CHmRun        0.198876945
## CRuns         0.053594554
## CRBI          0.055409116
## CWalks        0.054405147
## LeagueN       2.463634059
## DivisionW   -18.860043802
## PutOuts       0.046088692
## Assists       0.006674057
## Errors       -0.112913895
## NewLeagueN    2.358350957
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
## (Intercept) 240.682852262
## AtBat         0.083042199
## Hits          0.334990595
## HmRun         1.105720594
## Runs          0.542496738
## RBI           0.545363022
## Walks         0.698162048
## Years         2.316374820
## CAtBat        0.006988341
## CHits         0.026721778
## CHmRun        0.198876945
## CRuns         0.053594554
## CRBI          0.055409116
## CWalks        0.054405147
## LeagueN       2.463634059
## DivisionW   -18.860043802
## PutOuts       0.046088692
## Assists       0.006674057
## Errors       -0.112913895
## NewLeagueN    2.358350957
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 375.1832
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 136525.7
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.4731 450.0041 449.0536 448.7538 448.4658 448.1512 447.8075
##  [8] 447.4324 447.0230 446.5767 446.0902 445.5604 444.9838 444.3567
## [15] 443.6754 442.9357 442.1337 441.2649 440.3251 439.3097 438.2141
## [22] 437.0340 435.7649 434.4027 432.9433 431.3833 429.7194 427.9492
## [29] 426.0710 424.0839 421.9879 419.7844 417.4761 415.0668 412.5623
## [36] 409.9696 407.2977 404.5567 401.7586 398.9167 396.0458 393.1616
## [43] 390.2804 387.4191 384.5946 381.8234 379.1213 376.5029 373.9815
## [50] 371.5682 369.2728 367.1021 365.0624 363.1563 361.3850 359.7477
## [57] 358.2421 356.8643 355.6091 354.4707 353.4425 352.5175 351.6888
## [64] 350.9468 350.2895 349.7103 349.1998 348.7544 348.3613 348.0218
## [71] 347.7340 347.4863 347.2783 347.1055 346.9603 346.8446 346.7530
## [78] 346.6809 346.6260 346.5820 346.5501 346.5271 346.5098 346.4946
## [85] 346.4784 346.4688 346.4448 346.4298 346.3942 346.3678 346.3163
## [92] 346.2745 346.2065 346.1476 346.0635 345.9840 345.8839 345.7935
## [99] 345.6792
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 345.6792
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 376.5029
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
##                         1
## (Intercept)  153.26245052
## AtBat         -1.93059099
## Hits           6.80153148
## HmRun          1.30268871
## Runs          -1.03744367
## RBI            .         
## Walks          5.62556821
## Years         -7.58334496
## CAtBat        -0.06800322
## CHits          .         
## CHmRun         0.21277161
## CRuns          1.13540300
## CRBI           0.57729486
## CWalks        -0.73080821
## LeagueN       47.11254249
## DivisionW   -116.63369442
## PutOuts        0.28199476
## Assists        0.29084052
## Errors        -2.87461295
## NewLeagueN   -10.65739592
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 204.8565
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
##  [1] 450.1187 444.0050 434.0319 425.5996 418.0379 410.5305 402.3381
##  [8] 394.9802 388.7345 383.3894 378.8878 375.0111 371.7832 369.0828
## [15] 366.4147 363.1114 359.8716 357.1603 354.9205 353.0596 351.5155
## [22] 350.2345 349.1793 348.3131 347.5953 346.9974 346.5300 346.2828
## [29] 346.1376 346.0477 346.0042 345.9727 345.9580 346.0152 346.1847
## [36] 346.2768 346.5992 347.0462 347.1614 346.8893 346.0735 344.9782
## [43] 344.0729 343.2071 342.2955 341.6441 341.1911 340.8429 340.5035
## [50] 340.1732 340.0618 340.0684 340.1552 340.2385 340.3039 340.2913
## [57] 340.2282 340.1583 340.1070 340.0708 340.0470 340.0604 340.0898
## [64] 340.1165 340.1609 340.2082 340.2773 340.3203 340.3834 340.4469
## [71] 340.5324 340.4733 340.5809
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.047
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.0828
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
## 1  255.2820965 202606.8  38646.55  241253.4 163960.26     0
## 2  232.6035386 197140.5  39118.48  236259.0 158022.00     1
## 3  211.9396813 188383.7  37595.58  225979.3 150788.10     2
## 4  193.1115442 181135.0  36181.68  217316.7 144953.32     2
## 5  175.9560468 174755.6  34840.22  209595.9 139915.43     3
## 6  160.3245966 168535.3  33731.99  202267.3 134803.28     4
## 7  146.0818013 161875.9  32545.92  194421.9 129330.01     4
## 8  133.1042967 156009.3  31412.69  187422.0 124596.64     4
## 9  121.2796778 151114.5  30416.19  181530.7 120698.34     4
## 10 110.5055255 146987.4  29499.96  176487.4 117487.44     4
## 11 100.6885192 143556.0  28682.18  172238.2 114873.80     5
## 12  91.7436287 140633.3  27944.40  168577.7 112688.90     5
## 13  83.5933775 138222.8  27322.69  165545.5 110900.08     5
## 14  76.1671723 136222.1  26798.27  163020.4 109423.87     5
## 15  69.4006906 134259.8  26362.48  160622.2 107897.28     6
## 16  63.2353245 131849.9  25911.10  157761.0 105938.80     6
## 17  57.6176726 129507.6  25422.15  154929.7 104085.44     6
## 18  52.4990774 127563.5  25026.52  152590.0 102536.94     6
## 19  47.8352040 125968.6  24715.18  150683.8 101253.39     6
## 20  43.5856563 124651.1  24462.00  149113.1 100189.08     6
## 21  39.7136268 123563.1  24257.65  147820.8  99305.49     6
## 22  36.1855776 122664.2  24093.47  146757.7  98570.72     6
## 23  32.9709506 121926.2  23961.21  145887.4  97964.98     6
## 24  30.0419022 121322.0  23853.15  145175.1  97468.83     6
## 25  27.3730624 120822.5  23766.72  144589.2  97055.80     6
## 26  24.9413150 120407.2  23696.76  144103.9  96710.42     6
## 27  22.7255973 120083.1  23638.31  143721.4  96444.75     6
## 28  20.7067179 119911.8  23585.09  143496.8  96326.68     6
## 29  18.8671902 119811.3  23551.01  143362.3  96260.25     6
## 30  17.1910810 119749.0  23518.28  143267.3  96230.73     7
## 31  15.6638727 119718.9  23490.26  143209.2  96228.68     7
## 32  14.2723374 119697.1  23466.59  143163.7  96230.50     7
## 33  13.0044223 119687.0  23446.73  143133.7  96240.24     9
## 34  11.8491453 119726.5  23425.08  143151.6  96301.46     9
## 35  10.7964999 119843.8  23392.67  143236.5  96451.18     9
## 36   9.8373686 119907.6  23365.23  143272.8  96542.36     9
## 37   8.9634439 120131.0  23325.63  143456.6  96805.38     9
## 38   8.1671562 120441.1  23296.33  143737.4  97144.72    11
## 39   7.4416086 120521.0  23275.76  143796.8  97245.27    11
## 40   6.7805166 120332.2  23235.49  143567.7  97096.69    12
## 41   6.1781542 119766.9  23070.70  142837.6  96696.17    12
## 42   5.6293040 119010.0  22705.70  141715.7  96304.26    13
## 43   5.1292121 118386.1  22390.95  140777.1  95995.17    13
## 44   4.6735471 117791.1  22130.21  139921.3  95660.92    13
## 45   4.2583620 117166.2  21956.78  139123.0  95209.42    13
## 46   3.8800609 116720.7  21887.81  138608.5  94832.90    13
## 47   3.5353670 116411.4  21867.00  138278.4  94544.39    13
## 48   3.2212947 116173.9  21867.23  138041.1  94306.65    13
## 49   2.9351238 115942.7  21827.66  137770.3  94115.00    13
## 50   2.6743755 115717.8  21728.30  137446.1  93989.53    13
## 51   2.4367913 115642.0  21643.94  137286.0  93998.08    13
## 52   2.2203135 115646.5  21569.11  137215.6  94077.40    14
## 53   2.0230670 115705.6  21508.16  137213.7  94197.39    15
## 54   1.8433433 115762.3  21455.73  137218.0  94306.53    15
## 55   1.6795857 115806.8  21410.92  137217.7  94395.84    17
## 56   1.5303760 115798.2  21365.13  137163.3  94433.04    17
## 57   1.3944216 115755.2  21298.68  137053.9  94456.57    17
## 58   1.2705450 115707.7  21257.94  136965.6  94449.75    17
## 59   1.1576733 115672.7  21226.79  136899.5  94445.95    17
## 60   1.0548288 115648.2  21199.59  136847.8  94448.58    17
## 61   0.9611207 115631.9  21182.55  136814.5  94449.39    17
## 62   0.8757374 115641.1  21176.14  136817.2  94464.96    17
## 63   0.7979393 115661.0  21171.74  136832.8  94489.30    17
## 64   0.7270526 115679.3  21168.37  136847.6  94510.89    17
## 65   0.6624632 115709.4  21169.98  136879.4  94539.46    18
## 66   0.6036118 115741.6  21168.94  136910.5  94572.66    18
## 67   0.5499886 115788.7  21166.80  136955.5  94621.86    18
## 68   0.5011291 115817.9  21170.15  136988.1  94647.76    17
## 69   0.4566102 115860.9  21172.75  137033.6  94688.11    18
## 70   0.4160462 115904.1  21175.51  137079.6  94728.56    18
## 71   0.3790858 115962.3  21176.90  137139.2  94785.43    18
## 72   0.3454089 115922.1  21189.28  137111.4  94732.79    18
## 73   0.3147237 115995.3  21191.76  137187.1  94803.56    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1  0.9611207   76.16717
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

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.4.2 and the following packages:

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
## [61] "sfsmisc"      "parallel"     "survival"     "yaml"        
## [65] "colorspace"   "knitr"        "bindr"
```
