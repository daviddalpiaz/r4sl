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
##  [1] 450.7676 449.2877 448.4045 448.1396 447.8501 447.5338 447.1882
##  [8] 446.8110 446.3993 445.9504 445.4610 444.9281 444.3479 443.7170
## [15] 443.0313 442.2868 441.4793 440.6046 439.6581 438.6352 437.5313
## [22] 436.3419 435.0625 433.6886 432.2163 430.6418 428.9618 427.1736
## [29] 425.2753 423.2657 421.1448 418.9136 416.5745 414.1312 411.5891
## [36] 408.9550 406.2376 403.4467 400.5942 397.6932 394.7585 391.8054
## [43] 388.8506 385.9108 383.0031 380.1442 377.3503 374.6364 372.0161
## [50] 369.5013 367.1023 364.8264 362.6804 360.6679 358.7907 357.0485
## [57] 355.4394 353.9602 352.6061 351.3715 350.2500 349.2349 348.3191
## [64] 347.4948 346.7582 346.1014 345.5152 344.9938 344.5324 344.1245
## [71] 343.7681 343.4534 343.1766 342.9337 342.7222 342.5346 342.3681
## [78] 342.2223 342.0871 341.9668 341.8522 341.7442 341.6386 341.5353
## [85] 341.4286 341.3207 341.2097 341.0915 340.9676 340.8372 340.7012
## [92] 340.5574 340.4049 340.2460 340.0812 339.9125 339.7358 339.5571
## [99] 339.3747
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.3747
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.5013
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
## (Intercept) 2.220974e+02
## AtBat       .           
## Hits        1.129009e+00
## HmRun       .           
## Runs        .           
## RBI         .           
## Walks       1.172062e+00
## Years       .           
## CAtBat      .           
## CHits       .           
## CHmRun      .           
## CRuns       1.147170e-01
## CRBI        3.085475e-01
## CWalks      .           
## LeagueN     .           
## DivisionW   .           
## PutOuts     1.763115e-03
## Assists     .           
## Errors      .           
## NewLeagueN  .
```

```r
coef(fit_lasso_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept)  123.7520756
## AtBat         -1.5473426
## Hits           5.6608972
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.7296908
## Years         -9.5958375
## CAtBat         .        
## CHits          .        
## CHmRun         0.5108207
## CRuns          0.6594856
## CRBI           0.3927505
## CWalks        -0.5291586
## LeagueN       32.0650811
## DivisionW   -119.2990171
## PutOuts        0.2724045
## Assists        0.1732025
## Errors        -2.0585083
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 177.4942
```

```r
coef(fit_lasso_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 2.220974e+02
## AtBat       .           
## Hits        1.129009e+00
## HmRun       .           
## Runs        .           
## RBI         .           
## Walks       1.172062e+00
## Years       .           
## CAtBat      .           
## CHits       .           
## CHmRun      .           
## CRuns       1.147170e-01
## CRBI        3.085475e-01
## CWalks      .           
## LeagueN     .           
## DivisionW   .           
## PutOuts     1.763115e-03
## Assists     .           
## Errors      .           
## NewLeagueN  .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 2.726099
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 130946.2
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 451.5442 441.2321 430.8707 421.9137 413.2450 404.2229 396.0860
##  [8] 388.5717 381.9207 376.4239 371.7496 367.5462 363.9760 360.8332
## [15] 357.7240 354.7428 352.0125 349.6168 347.5550 345.8860 344.5119
## [22] 343.3849 342.4787 341.7334 341.1377 340.6567 340.3900 340.3702
## [29] 340.4648 340.6248 340.7623 340.8972 341.0342 341.2430 341.5978
## [36] 342.0353 342.6223 343.1905 343.6310 343.6173 342.9203 342.1088
## [43] 341.3356 340.6680 339.9201 339.2781 338.7246 338.3270 338.0091
## [50] 337.8806 337.9484 338.1204 338.3420 338.5738 338.8081 338.9964
## [57] 339.2120 339.3933 339.5295 339.6333 339.7821 339.9412 340.1052
## [64] 340.2683 340.4458 340.6119 340.8129 340.9460 341.1139 341.1987
## [71] 341.3780 341.4579 341.5878
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.8806
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.7496
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
## 1  255.2820965 203892.1  37236.35  241128.5 166655.77     0
## 2  232.6035386 194685.7  36691.92  231377.7 157993.82     1
## 3  211.9396813 185649.6  35732.21  221381.8 149917.39     2
## 4  193.1115442 178011.2  34943.53  212954.7 143067.67     2
## 5  175.9560468 170771.5  34324.25  205095.7 136447.21     3
## 6  160.3245966 163396.2  33972.48  197368.7 129423.69     4
## 7  146.0818013 156884.1  33638.42  190522.5 123245.68     4
## 8  133.1042967 150988.0  33205.70  184193.7 117782.30     4
## 9  121.2796778 145863.4  32779.14  178642.5 113084.26     4
## 10 110.5055255 141694.9  32460.71  174155.7 109234.23     4
## 11 100.6885192 138197.8  32209.87  170407.6 105987.91     5
## 12  91.7436287 135090.2  32010.89  167101.1 103079.33     5
## 13  83.5933775 132478.6  31861.76  164340.3 100616.80     5
## 14  76.1671723 130200.6  31708.02  161908.6  98492.59     5
## 15  69.4006906 127966.4  31572.59  159539.0  96393.84     6
## 16  63.2353245 125842.4  31429.78  157272.2  94412.64     6
## 17  57.6176726 123912.8  31231.28  155144.1  92681.53     6
## 18  52.4990774 122231.9  30975.31  153207.2  91256.62     6
## 19  47.8352040 120794.5  30704.88  151499.4  90089.60     6
## 20  43.5856563 119637.1  30473.85  150111.0  89163.30     6
## 21  39.7136268 118688.4  30269.18  148957.6  88419.25     6
## 22  36.1855776 117913.2  30088.52  148001.7  87824.67     6
## 23  32.9709506 117291.6  29927.82  147219.5  87363.83     6
## 24  30.0419022 116781.7  29783.98  146565.7  86997.71     6
## 25  27.3730624 116374.9  29661.08  146036.0  86713.85     6
## 26  24.9413150 116047.0  29550.54  145597.5  86496.44     6
## 27  22.7255973 115865.3  29474.67  145340.0  86390.67     6
## 28  20.7067179 115851.8  29452.57  145304.4  86399.27     6
## 29  18.8671902 115916.3  29448.05  145364.3  86468.24     6
## 30  17.1910810 116025.2  29441.28  145466.5  86583.94     7
## 31  15.6638727 116119.0  29433.10  145552.1  86685.88     7
## 32  14.2723374 116210.9  29423.24  145634.2  86787.68     7
## 33  13.0044223 116304.3  29411.30  145715.6  86893.04     9
## 34  11.8491453 116446.8  29416.92  145863.7  87029.85     9
## 35  10.7964999 116689.1  29452.46  146141.5  87236.61     9
## 36   9.8373686 116988.1  29486.95  146475.1  87501.16     9
## 37   8.9634439 117390.1  29508.36  146898.4  87881.70     9
## 38   8.1671562 117779.8  29518.60  147298.4  88261.15    11
## 39   7.4416086 118082.3  29518.83  147601.1  88563.46    11
## 40   6.7805166 118072.9  29473.43  147546.3  88599.44    12
## 41   6.1781542 117594.3  29274.97  146869.3  88319.34    12
## 42   5.6293040 117038.4  28944.76  145983.2  88093.65    13
## 43   5.1292121 116510.0  28593.21  145103.2  87916.81    13
## 44   4.6735471 116054.7  28273.80  144328.5  87780.88    13
## 45   4.2583620 115545.7  27930.19  143475.9  87615.51    13
## 46   3.8800609 115109.7  27612.28  142721.9  87497.38    13
## 47   3.5353670 114734.4  27330.42  142064.8  87403.96    13
## 48   3.2212947 114465.1  27088.04  141553.2  87377.10    13
## 49   2.9351238 114250.2  26865.08  141115.2  87385.07    13
## 50   2.6743755 114163.3  26656.19  140819.5  87507.14    13
## 51   2.4367913 114209.1  26488.08  140697.2  87721.01    13
## 52   2.2203135 114325.4  26356.36  140681.7  87969.02    14
## 53   2.0230670 114475.3  26236.56  140711.9  88238.77    15
## 54   1.8433433 114632.2  26126.13  140758.3  88506.08    15
## 55   1.6795857 114790.9  26036.69  140827.6  88754.22    17
## 56   1.5303760 114918.6  25962.84  140881.4  88955.72    17
## 57   1.3944216 115064.7  25937.99  141002.7  89126.76    17
## 58   1.2705450 115187.8  25939.42  141127.2  89248.40    17
## 59   1.1576733 115280.3  25922.47  141202.8  89357.81    17
## 60   1.0548288 115350.8  25905.77  141256.6  89445.02    17
## 61   0.9611207 115451.9  25904.90  141356.8  89546.98    17
## 62   0.8757374 115560.0  25910.81  141470.8  89649.22    17
## 63   0.7979393 115671.5  25923.28  141594.8  89748.26    17
## 64   0.7270526 115782.5  25940.34  141722.8  89842.17    17
## 65   0.6624632 115903.4  25955.58  141858.9  89947.78    18
## 66   0.6036118 116016.5  25970.97  141987.5  90045.52    18
## 67   0.5499886 116153.4  25990.67  142144.1  90162.77    18
## 68   0.5011291 116244.2  26003.35  142247.5  90240.82    17
## 69   0.4566102 116358.7  26019.69  142378.4  90339.00    18
## 70   0.4160462 116416.5  26031.50  142448.0  90385.04    18
## 71   0.3790858 116538.9  26044.06  142583.0  90494.85    18
## 72   0.3454089 116593.5  26054.44  142647.9  90539.06    18
## 73   0.3147237 116682.2  26068.12  142750.3  90614.11    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   100.6885
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
