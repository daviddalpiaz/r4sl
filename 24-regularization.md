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
##  [1] 451.8048 450.1272 449.2661 449.0007 448.7105 448.3934 448.0471
##  [8] 447.6690 447.2564 446.8064 446.3159 445.7816 445.2001 444.5675
## [15] 443.8801 443.1336 442.3240 441.4469 440.4977 439.4718 438.3647
## [22] 437.1716 435.8881 434.5098 433.0324 431.4523 429.7661 427.9711
## [29] 426.0652 424.0472 421.9170 419.6756 417.3252 414.8697 412.3141
## [36] 409.6653 406.9320 404.1241 401.2532 398.3327 395.3773 392.4025
## [43] 389.4251 386.4621 383.5307 380.6480 377.8305 375.0936 372.4513
## [50] 369.9163 367.4985 365.2071 363.0487 361.0275 359.1458 357.4040
## [57] 355.8007 354.3329 352.9964 351.7858 350.6950 349.7173 348.8459
## [64] 348.0727 347.3945 346.8026 346.2881 345.8451 345.4673 345.1511
## [71] 344.8915 344.6787 344.5090 344.3780 344.2832 344.2181 344.1764
## [78] 344.1587 344.1577 344.1720 344.1969 344.2296 344.2668 344.3100
## [85] 344.3520 344.3923 344.4313 344.4659 344.4941 344.5149 344.5311
## [92] 344.5381 344.5374 344.5297 344.5146 344.4933 344.4649 344.4314
## [99] 344.3932
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.1577
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.9163
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
## (Intercept)  149.56860909
## AtBat         -1.88405954
## Hits           6.56495714
## HmRun          0.89198861
## Runs          -0.60661686
## RBI            .         
## Walks          5.44091255
## Years         -8.55574144
## CAtBat        -0.04537933
## CHits          .         
## CHmRun         0.34629402
## CRuns          1.00226519
## CRBI           0.51090279
## CWalks        -0.69507208
## LeagueN       42.48466139
## DivisionW   -117.38329115
## PutOuts        0.28121467
## Assists        0.26787019
## Errors        -2.72019905
## NewLeagueN    -6.64596717
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 196.3274
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
##  [1] 450.9378 440.8872 430.6274 421.8747 413.5832 404.8706 396.4892
##  [8] 389.1140 382.9591 377.5513 372.9356 368.7217 364.8092 361.2035
## [15] 357.6533 354.3092 351.3099 348.6915 346.4803 344.6423 343.0975
## [22] 341.8093 340.7815 340.0276 339.4893 339.0683 338.7489 338.5229
## [29] 338.4235 338.3708 338.3456 338.3460 338.3547 338.3624 338.4024
## [36] 338.4063 338.6179 338.9480 339.0534 338.5981 337.7878 336.8876
## [43] 336.1691 335.5314 334.9178 334.2734 333.7101 333.2366 332.8415
## [50] 332.5320 332.3395 332.2527 332.2313 332.2983 332.3000 332.2266
## [57] 332.1869 332.1818 332.2334 332.3333 332.4247 332.4969 332.5588
## [64] 332.6336 332.7043 332.7750 332.8402 332.9080 332.9924 333.0689
## [71] 333.1715 333.2661 333.3484 333.4043
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 332.1818
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 357.6533
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
## 1  255.2820965 203344.9  23754.87  227099.8 179590.03     0
## 2  232.6035386 194381.5  23821.43  218202.9 170560.08     1
## 3  211.9396813 185440.0  23311.67  208751.7 162128.32     2
## 4  193.1115442 177978.3  22893.23  200871.5 155085.07     2
## 5  175.9560468 171051.0  22697.22  193748.2 148353.81     3
## 6  160.3245966 163920.2  22515.03  186435.2 141405.16     4
## 7  146.0818013 157203.7  22174.65  179378.3 135029.05     4
## 8  133.1042967 151409.7  21877.51  173287.2 129532.18     4
## 9  121.2796778 146657.6  21599.28  168256.9 125058.36     4
## 10 110.5055255 142545.0  21354.55  163899.6 121190.46     4
## 11 100.6885192 139081.0  21163.17  160244.1 117917.81     5
## 12  91.7436287 135955.7  21061.39  157017.1 114894.31     5
## 13  83.5933775 133085.7  20974.35  154060.1 112111.40     5
## 14  76.1671723 130468.0  20887.81  151355.8 109580.16     5
## 15  69.4006906 127915.9  20809.96  148725.9 107105.94     6
## 16  63.2353245 125535.0  20740.56  146275.6 104794.47     6
## 17  57.6176726 123418.7  20653.19  144071.9 102765.48     6
## 18  52.4990774 121585.7  20540.02  142125.8 101045.73     6
## 19  47.8352040 120048.6  20432.27  140480.9  99616.33     6
## 20  43.5856563 118778.3  20340.50  139118.8  98437.80     6
## 21  39.7136268 117715.9  20263.10  137979.0  97452.82     6
## 22  36.1855776 116833.6  20197.20  137030.8  96636.39     6
## 23  32.9709506 116132.0  20144.09  136276.1  95987.93     6
## 24  30.0419022 115618.8  20099.36  135718.1  95519.40     6
## 25  27.3730624 115253.0  20060.71  135313.7  95192.30     6
## 26  24.9413150 114967.3  20029.56  134996.9  94937.76     6
## 27  22.7255973 114750.8  20005.13  134755.9  94745.68     6
## 28  20.7067179 114597.8  19986.46  134584.2  94611.33     6
## 29  18.8671902 114530.5  19968.23  134498.7  94562.26     6
## 30  17.1910810 114494.8  19953.98  134448.8  94540.79     7
## 31  15.6638727 114477.7  19943.17  134420.9  94534.55     7
## 32  14.2723374 114478.0  19936.08  134414.1  94541.91     7
## 33  13.0044223 114483.9  19930.77  134414.7  94553.13     9
## 34  11.8491453 114489.1  19928.31  134417.4  94560.78     9
## 35  10.7964999 114516.2  19929.56  134445.7  94586.62     9
## 36   9.8373686 114518.8  19946.44  134465.2  94572.36     9
## 37   8.9634439 114662.1  19933.03  134595.1  94729.03     9
## 38   8.1671562 114885.8  19898.08  134783.8  94987.69    11
## 39   7.4416086 114957.2  19806.66  134763.8  95150.53    11
## 40   6.7805166 114648.7  19668.67  134317.4  94980.02    12
## 41   6.1781542 114100.6  19529.69  133630.3  94570.93    12
## 42   5.6293040 113493.2  19373.95  132867.2  94119.27    13
## 43   5.1292121 113009.7  19220.46  132230.1  93789.22    13
## 44   4.6735471 112581.3  19099.66  131681.0  93481.63    13
## 45   4.2583620 112169.9  18993.75  131163.7  93176.20    13
## 46   3.8800609 111738.7  18880.38  130619.1  92858.36    13
## 47   3.5353670 111362.4  18772.40  130134.8  92590.02    13
## 48   3.2212947 111046.6  18654.08  129700.7  92392.55    13
## 49   2.9351238 110783.5  18533.95  129317.4  92249.52    13
## 50   2.6743755 110577.6  18425.77  129003.3  92151.79    13
## 51   2.4367913 110449.6  18327.84  128777.4  92121.73    13
## 52   2.2203135 110391.8  18243.37  128635.2  92148.47    14
## 53   2.0230670 110377.7  18170.25  128547.9  92207.41    15
## 54   1.8433433 110422.2  18100.14  128522.3  92322.01    15
## 55   1.6795857 110423.3  18030.89  128454.2  92392.42    17
## 56   1.5303760 110374.5  17959.87  128334.4  92414.65    17
## 57   1.3944216 110348.1  17906.68  128254.8  92441.46    17
## 58   1.2705450 110344.8  17870.26  128215.0  92474.51    17
## 59   1.1576733 110379.0  17841.58  128220.6  92537.43    17
## 60   1.0548288 110445.4  17818.69  128264.1  92626.73    17
## 61   0.9611207 110506.1  17798.19  128304.3  92707.96    17
## 62   0.8757374 110554.2  17786.97  128341.1  92767.20    17
## 63   0.7979393 110595.4  17783.36  128378.7  92812.03    17
## 64   0.7270526 110645.1  17780.61  128425.7  92864.51    17
## 65   0.6624632 110692.2  17774.83  128467.0  92917.34    18
## 66   0.6036118 110739.2  17770.82  128510.0  92968.39    18
## 67   0.5499886 110782.6  17772.35  128554.9  93010.23    18
## 68   0.5011291 110827.7  17769.46  128597.2  93058.25    17
## 69   0.4566102 110884.0  17772.23  128656.2  93111.74    18
## 70   0.4160462 110934.9  17781.38  128716.3  93153.49    18
## 71   0.3790858 111003.3  17788.37  128791.7  93214.90    18
## 72   0.3454089 111066.3  17797.19  128863.5  93269.08    18
## 73   0.3147237 111121.2  17798.92  128920.1  93322.25    18
## 74   0.2867645 111158.4  17801.89  128960.3  93356.55    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   1.270545   69.40069
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
