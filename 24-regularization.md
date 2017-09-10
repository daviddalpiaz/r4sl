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
##  [1] 452.4641 450.5693 449.9942 449.7259 449.4326 449.1121 448.7620
##  [8] 448.3798 447.9628 447.5081 447.0125 446.4726 445.8851 445.2461
## [15] 444.5517 443.7978 442.9803 442.0948 441.1366 440.1012 438.9840
## [22] 437.7804 436.4859 435.0960 433.6068 432.0145 430.3159 428.5083
## [29] 426.5898 424.5594 422.4172 420.1642 417.8031 415.3377 412.7735
## [36] 410.1178 407.3794 404.5684 401.6968 398.7784 395.8278 392.8611
## [43] 389.8948 386.9463 384.0327 381.1711 378.3777 375.6679 373.0554
## [50] 370.5520 368.1680 365.9119 363.7895 361.8048 359.9595 358.2536
## [57] 356.6852 355.2512 353.9468 352.7666 351.7044 350.7535 349.9068
## [64] 349.1565 348.4995 347.9274 347.4321 347.0063 346.6456 346.3433
## [71] 346.1000 345.9011 345.7463 345.6324 345.5494 345.4971 345.4705
## [78] 345.4634 345.4740 345.4974 345.5321 345.5717 345.6156 345.6580
## [85] 345.6999 345.7376 345.7671 345.7901 345.8045 345.8072 345.7988
## [92] 345.7791 345.7481 345.7045 345.6520 345.5880 345.5152 345.4338
## [99] 345.3469
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 345.3469
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 378.3777
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
##                         1
## (Intercept)  1.395005e+02
## AtBat       -1.734125e+00
## Hits         6.059861e+00
## HmRun        1.708102e-01
## Runs         .           
## RBI          .           
## Walks        5.053826e+00
## Years       -1.061379e+01
## CAtBat      -2.483638e-05
## CHits        .           
## CHmRun       5.721414e-01
## CRuns        7.218106e-01
## CRBI         3.856339e-01
## CWalks      -6.033666e-01
## LeagueN      3.339735e+01
## DivisionW   -1.193924e+02
## PutOuts      2.772560e-01
## Assists      2.104828e-01
## Errors      -2.371468e+00
## NewLeagueN   .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 181.5643
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
##  [1] 450.1669 441.8438 431.6726 423.0547 415.2999 406.9272 398.6274
##  [8] 391.1441 384.9607 379.6988 375.3131 371.6268 368.7610 366.2239
## [15] 363.4222 360.5948 357.8711 355.4286 353.4087 351.7475 350.3843
## [22] 349.2625 348.3441 347.5857 346.9763 346.4739 346.0871 345.8140
## [29] 345.5972 345.4297 345.3530 345.2910 345.2081 345.1494 345.1572
## [36] 345.4350 345.8448 346.2011 346.3993 345.9591 344.9893 344.0021
## [43] 343.1058 342.3503 341.6802 341.1629 340.7812 340.4907 340.2625
## [50] 340.0711 339.9525 339.8496 339.8230 339.9225 339.9594 339.9962
## [57] 340.1025 340.1994 340.3220 340.4667 340.6876 340.8811 341.0772
## [64] 341.2598 341.3993 341.5359 341.6443 341.7529 341.8685 341.9803
## [71] 342.1495 342.2537 342.3129
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.823
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 363.4222
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
## 1  255.2820965 202650.3  21737.33  224387.6 180912.95     0
## 2  232.6035386 195226.0  21698.29  216924.3 173527.69     1
## 3  211.9396813 186341.2  20138.37  206479.6 166202.85     2
## 4  193.1115442 178975.2  18762.06  197737.3 160213.18     2
## 5  175.9560468 172474.0  17637.82  190111.8 154836.15     3
## 6  160.3245966 165589.8  16724.27  182314.0 148865.50     4
## 7  146.0818013 158903.8  15784.93  174688.7 143118.86     4
## 8  133.1042967 152993.7  15089.82  168083.5 137903.86     4
## 9  121.2796778 148194.7  14620.38  162815.1 133574.35     4
## 10 110.5055255 144171.2  14284.78  158455.9 129886.37     4
## 11 100.6885192 140859.9  14143.57  155003.5 126716.31     5
## 12  91.7436287 138106.5  14139.62  152246.1 123966.85     5
## 13  83.5933775 135984.7  14214.39  150199.0 121770.26     5
## 14  76.1671723 134120.0  14347.02  148467.0 119772.95     5
## 15  69.4006906 132075.7  14479.58  146555.3 117596.12     6
## 16  63.2353245 130028.6  14595.70  144624.3 115432.90     6
## 17  57.6176726 128071.7  14695.66  142767.4 113376.06     6
## 18  52.4990774 126329.5  14790.26  141119.7 111539.21     6
## 19  47.8352040 124897.7  14904.16  139801.9 109993.54     6
## 20  43.5856563 123726.3  15028.82  138755.1 108697.49     6
## 21  39.7136268 122769.2  15159.77  137929.0 107609.41     6
## 22  36.1855776 121984.3  15292.32  137276.6 106691.98     6
## 23  32.9709506 121343.6  15422.95  136766.6 105920.66     6
## 24  30.0419022 120815.8  15548.99  136364.8 105266.86     6
## 25  27.3730624 120392.6  15668.75  136061.3 104723.82     6
## 26  24.9413150 120044.2  15782.97  135827.1 104261.18     6
## 27  22.7255973 119776.3  15888.17  135664.5 103888.14     6
## 28  20.7067179 119587.3  15980.89  135568.2 103606.43     6
## 29  18.8671902 119437.5  16066.84  135504.3 103370.61     6
## 30  17.1910810 119321.7  16148.02  135469.7 103173.69     7
## 31  15.6638727 119268.7  16229.93  135498.6 103038.74     7
## 32  14.2723374 119225.9  16301.40  135527.3 102924.50     7
## 33  13.0044223 119168.6  16359.14  135527.8 102809.48     9
## 34  11.8491453 119128.1  16407.67  135535.7 102720.40     9
## 35  10.7964999 119133.5  16446.03  135579.5 102687.48     9
## 36   9.8373686 119325.4  16491.49  135816.8 102833.87     9
## 37   8.9634439 119608.6  16567.13  136175.8 103041.50     9
## 38   8.1671562 119855.2  16684.38  136539.6 103170.78    11
## 39   7.4416086 119992.5  16767.19  136759.7 103225.28    11
## 40   6.7805166 119687.7  16776.14  136463.8 102911.56    12
## 41   6.1781542 119017.6  16800.22  135817.8 102217.40    12
## 42   5.6293040 118337.4  16895.02  135232.4 101442.39    13
## 43   5.1292121 117721.6  16998.59  134720.2 100722.99    13
## 44   4.6735471 117203.7  17069.20  134272.9 100134.51    13
## 45   4.2583620 116745.4  17120.56  133865.9  99624.82    13
## 46   3.8800609 116392.1  17176.40  133568.5  99215.73    13
## 47   3.5353670 116131.8  17226.41  133358.2  98905.40    13
## 48   3.2212947 115933.9  17264.21  133198.1  98669.72    13
## 49   2.9351238 115778.6  17292.63  133071.2  98485.95    13
## 50   2.6743755 115648.4  17312.92  132961.3  98335.46    13
## 51   2.4367913 115567.7  17331.33  132899.0  98236.38    13
## 52   2.2203135 115497.8  17353.01  132850.8  98144.74    14
## 53   2.0230670 115479.7  17393.55  132873.2  98086.15    15
## 54   1.8433433 115547.3  17421.05  132968.4  98126.27    15
## 55   1.6795857 115572.4  17450.61  133023.0  98121.82    17
## 56   1.5303760 115597.4  17470.65  133068.1  98126.80    17
## 57   1.3944216 115669.7  17484.93  133154.7  98184.80    17
## 58   1.2705450 115735.6  17524.43  133260.0  98211.18    17
## 59   1.1576733 115819.0  17568.25  133387.3  98250.79    17
## 60   1.0548288 115917.6  17611.56  133529.1  98306.02    17
## 61   0.9611207 116068.0  17656.28  133724.3  98411.77    17
## 62   0.8757374 116199.9  17698.90  133898.8  98501.03    17
## 63   0.7979393 116333.7  17739.26  134072.9  98594.40    17
## 64   0.7270526 116458.3  17786.62  134244.9  98671.65    17
## 65   0.6624632 116553.4  17828.37  134381.8  98725.08    18
## 66   0.6036118 116646.7  17868.90  134515.6  98777.84    18
## 67   0.5499886 116720.8  17903.65  134624.4  98817.15    18
## 68   0.5011291 116795.0  17939.68  134734.7  98855.34    17
## 69   0.4566102 116874.1  17966.71  134840.8  98907.38    18
## 70   0.4160462 116950.6  17985.09  134935.6  98965.47    18
## 71   0.3790858 117066.3  18022.53  135088.8  99043.74    18
## 72   0.3454089 117137.6  18038.27  135175.9  99099.33    18
## 73   0.3147237 117178.2  18060.17  135238.3  99117.99    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.023067   69.40069
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
