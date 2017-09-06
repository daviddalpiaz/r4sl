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
## (Intercept) 226.844379273
## AtBat         0.086613903
## Hits          0.352962516
## HmRun         1.144213853
## Runs          0.569353374
## RBI           0.570074068
## Walks         0.735072620
## Years         2.397356093
## CAtBat        0.007295083
## CHits         0.027995153
## CHmRun        0.208112350
## CRuns         0.056146220
## CRBI          0.058060281
## CWalks        0.056586702
## LeagueN       2.850306112
## DivisionW   -20.329125702
## PutOuts       0.049296951
## Assists       0.007063169
## Errors       -0.128066381
## NewLeagueN    2.654025563
```

```r
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)   71.78758429
## AtBat         -0.58269657
## Hits           2.51715272
## HmRun         -1.39973428
## Runs           1.07259572
## RBI            0.74825248
## Walks          3.17950553
## Years         -8.35976899
## CAtBat         0.00133718
## CHits          0.12772556
## CHmRun         0.68074413
## CRuns          0.27080732
## CRBI           0.24581306
## CWalks        -0.24120197
## LeagueN       51.41107146
## DivisionW   -121.93563378
## PutOuts        0.26073685
## Assists        0.15595798
## Errors        -3.59749877
## NewLeagueN   -15.89754187
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 17868.18
```

```r
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 226.844379273
## AtBat         0.086613903
## Hits          0.352962516
## HmRun         1.144213853
## Runs          0.569353374
## RBI           0.570074068
## Walks         0.735072620
## Years         2.397356093
## CAtBat        0.007295083
## CHits         0.027995153
## CHmRun        0.208112350
## CRuns         0.056146220
## CRBI          0.058060281
## CWalks        0.056586702
## LeagueN       2.850306112
## DivisionW   -20.329125702
## PutOuts       0.049296951
## Assists       0.007063169
## Errors       -0.128066381
## NewLeagueN    2.654025563
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 436.8923
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 134397.5
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.3221 449.6993 448.9212 448.6538 448.3615 448.0421 447.6932
##  [8] 447.3124 446.8968 446.4435 445.9496 445.4115 444.8259 444.1890
## [15] 443.4969 442.7455 441.9306 441.0478 440.0926 439.0604 437.9466
## [22] 436.7466 435.4558 434.0700 432.5850 430.9970 429.3029 427.4999
## [29] 425.5861 423.5605 421.4230 419.1748 416.8183 414.3573 411.7973
## [36] 409.1455 406.4105 403.6023 400.7330 397.8160 394.8662 391.8992
## [43] 388.9318 385.9810 383.0641 380.1979 377.3988 374.6819 372.0613
## [50] 369.5485 367.1538 364.8846 362.7475 360.7467 358.8834 357.1574
## [57] 355.5666 354.1076 352.7756 351.5649 350.4690 349.4811 348.5936
## [64] 347.7951 347.0873 346.4623 345.9079 345.4171 344.9898 344.6185
## [71] 344.3003 344.0157 343.7781 343.5770 343.4004 343.2569 343.1300
## [78] 343.0300 342.9369 342.8646 342.7954 342.7349 342.6832 342.6232
## [85] 342.5702 342.5126 342.4534 342.3916 342.3218 342.2439 342.1629
## [92] 342.0724 341.9782 341.8747 341.7689 341.6546 341.5399 341.4191
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.4191
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 372.0613
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
## (Intercept)  1.448326e+02
## AtBat       -1.822945e+00
## Hits         6.240387e+00
## HmRun        3.286987e-01
## Runs        -7.473816e-03
## RBI          .           
## Walks        5.189642e+00
## Years       -9.819653e+00
## CAtBat      -1.554270e-02
## CHits        .           
## CHmRun       5.229006e-01
## CRuns        8.252401e-01
## CRBI         4.243410e-01
## CWalks      -6.475166e-01
## LeagueN      3.571403e+01
## DivisionW   -1.183448e+02
## PutOuts      2.803403e-01
## Assists      2.379593e-01
## Errors      -2.513172e+00
## NewLeagueN  -7.350528e-01
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 183.6697
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
##  [1] 451.2982 442.2651 432.1765 423.6278 415.5767 406.4964 397.5364
##  [8] 389.7075 383.2072 377.7997 373.1876 369.1245 365.6821 362.5910
## [15] 359.5282 356.6235 354.0328 351.6190 349.4996 347.7320 346.2602
## [22] 345.0358 344.0225 343.1878 342.5035 341.9810 341.6041 341.3363
## [29] 341.1495 341.0580 340.9976 340.9317 340.8486 340.8801 341.2045
## [36] 341.6037 341.8819 341.9261 341.9338 341.8885 341.8406 341.6694
## [43] 341.2568 340.6891 340.0312 339.2611 338.6307 338.1525 337.8251
## [50] 337.6942 337.5794 337.5161 337.4794 337.4025 337.3615 337.3775
## [57] 337.4157 337.4812 337.5766 337.5741 337.6172 337.6708 337.7414
## [64] 337.8090 337.8665 337.9290 338.0056 338.0855 338.1723 338.2505
## [71] 338.3562 338.4024 338.5215 338.6194
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.3615
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 362.591
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
## 1  255.2820965 203670.0  28787.34  232457.4 174882.68     0
## 2  232.6035386 195598.5  29014.17  224612.6 166584.28     1
## 3  211.9396813 186776.5  28032.10  214808.6 158744.44     2
## 4  193.1115442 179460.5  27106.71  206567.2 152353.80     2
## 5  175.9560468 172704.0  26210.17  198914.1 146493.79     3
## 6  160.3245966 165239.4  25536.88  190776.2 139702.48     4
## 7  146.0818013 158035.2  24970.93  183006.2 133064.30     4
## 8  133.1042967 151871.9  24476.37  176348.3 127395.57     4
## 9  121.2796778 146847.7  24072.76  170920.5 122774.96     4
## 10 110.5055255 142732.6  23746.65  166479.2 118985.95     4
## 11 100.6885192 139269.0  23465.72  162734.7 115803.24     5
## 12  91.7436287 136252.9  23196.21  159449.1 113056.70     5
## 13  83.5933775 133723.4  23002.36  156725.8 110721.05     5
## 14  76.1671723 131472.2  22862.72  154334.9 108609.49     5
## 15  69.4006906 129260.5  22797.36  152057.9 106463.13     6
## 16  63.2353245 127180.3  22749.28  149929.6 104431.07     6
## 17  57.6176726 125339.2  22639.12  147978.3 102700.09     6
## 18  52.4990774 123635.9  22454.43  146090.3 101181.45     6
## 19  47.8352040 122150.0  22273.49  144423.5  99876.51     6
## 20  43.5856563 120917.5  22122.72  143040.3  98794.83     6
## 21  39.7136268 119896.1  21998.39  141894.5  97897.71     6
## 22  36.1855776 119049.7  21894.85  140944.5  97154.84     6
## 23  32.9709506 118351.5  21808.91  140160.4  96542.56     6
## 24  30.0419022 117777.8  21739.27  139517.1  96038.58     6
## 25  27.3730624 117308.6  21682.26  138990.9  95626.37     6
## 26  24.9413150 116951.0  21635.01  138586.0  95316.00     6
## 27  22.7255973 116693.4  21595.20  138288.6  95098.15     6
## 28  20.7067179 116510.4  21561.69  138072.1  94948.75     6
## 29  18.8671902 116383.0  21532.19  137915.2  94850.81     6
## 30  17.1910810 116320.5  21500.38  137820.9  94820.15     7
## 31  15.6638727 116279.4  21474.26  137753.6  94805.12     7
## 32  14.2723374 116234.4  21452.91  137687.3  94781.51     7
## 33  13.0044223 116177.8  21432.81  137610.6  94744.98     9
## 34  11.8491453 116199.3  21395.86  137595.1  94803.40     9
## 35  10.7964999 116420.5  21314.00  137734.5  95106.49     9
## 36   9.8373686 116693.1  21237.64  137930.7  95455.43     9
## 37   8.9634439 116883.2  21176.30  138059.5  95706.93     9
## 38   8.1671562 116913.5  21156.17  138069.6  95757.28    11
## 39   7.4416086 116918.7  21145.47  138064.2  95773.27    11
## 40   6.7805166 116887.7  21074.94  137962.7  95812.79    12
## 41   6.1781542 116855.0  20980.82  137835.8  95874.20    12
## 42   5.6293040 116737.9  20840.62  137578.6  95897.32    13
## 43   5.1292121 116456.2  20633.54  137089.8  95822.69    13
## 44   4.6735471 116069.1  20374.93  136444.0  95694.14    13
## 45   4.2583620 115621.2  20118.80  135740.0  95502.43    13
## 46   3.8800609 115098.1  19844.79  134942.9  95253.31    13
## 47   3.5353670 114670.7  19578.10  134248.8  95092.64    13
## 48   3.2212947 114347.1  19343.62  133690.7  95003.47    13
## 49   2.9351238 114125.8  19132.10  133257.9  94993.73    13
## 50   2.6743755 114037.4  18925.30  132962.7  95112.07    13
## 51   2.4367913 113959.8  18728.20  132688.1  95231.65    13
## 52   2.2203135 113917.1  18552.01  132469.1  95365.11    14
## 53   2.0230670 113892.3  18384.19  132276.5  95508.12    15
## 54   1.8433433 113840.5  18229.82  132070.3  95610.66    15
## 55   1.6795857 113812.8  18101.36  131914.1  95711.41    17
## 56   1.5303760 113823.6  17990.74  131814.3  95832.81    17
## 57   1.3944216 113849.4  17891.17  131740.5  95958.20    17
## 58   1.2705450 113893.5  17805.01  131698.5  96088.52    17
## 59   1.1576733 113958.0  17729.41  131687.4  96228.57    17
## 60   1.0548288 113956.3  17688.41  131644.7  96267.89    17
## 61   0.9611207 113985.3  17647.14  131632.5  96338.21    17
## 62   0.8757374 114021.6  17607.17  131628.7  96414.38    17
## 63   0.7979393 114069.3  17568.28  131637.5  96500.97    17
## 64   0.7270526 114114.9  17531.38  131646.3  96583.54    17
## 65   0.6624632 114153.8  17497.19  131651.0  96656.59    18
## 66   0.6036118 114196.0  17467.33  131663.4  96728.70    18
## 67   0.5499886 114247.8  17440.47  131688.3  96807.35    18
## 68   0.5011291 114301.8  17419.80  131721.6  96882.02    17
## 69   0.4566102 114360.5  17395.10  131755.6  96965.42    18
## 70   0.4160462 114413.4  17379.27  131792.6  97034.10    18
## 71   0.3790858 114484.9  17371.66  131856.6  97113.26    18
## 72   0.3454089 114516.2  17363.77  131879.9  97152.40    18
## 73   0.3147237 114596.8  17360.08  131956.9  97236.71    18
## 74   0.2867645 114663.1  17357.64  132020.8  97305.48    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   1.679586   76.16717
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
##  [1] "purrr"        "reshape2"     "splines"      "colorspace"  
##  [5] "htmltools"    "stats4"       "yaml"         "mgcv"        
##  [9] "rlang"        "e1071"        "ModelMetrics" "nloptr"      
## [13] "foreign"      "glue"         "bindrcpp"     "bindr"       
## [17] "plyr"         "stringr"      "MatrixModels" "munsell"     
## [21] "gtable"       "codetools"    "psych"        "evaluate"    
## [25] "knitr"        "SparseM"      "class"        "quantreg"    
## [29] "pbkrtest"     "parallel"     "Rcpp"         "backports"   
## [33] "scales"       "lme4"         "mnormt"       "digest"      
## [37] "stringi"      "bookdown"     "dplyr"        "grid"        
## [41] "rprojroot"    "tools"        "magrittr"     "lazyeval"    
## [45] "tibble"       "tidyr"        "car"          "pkgconfig"   
## [49] "MASS"         "assertthat"   "minqa"        "rmarkdown"   
## [53] "iterators"    "R6"           "nnet"         "nlme"        
## [57] "compiler"
```
