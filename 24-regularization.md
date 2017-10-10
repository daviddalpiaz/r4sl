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
##  [1] 450.2894 448.9533 448.0004 447.7328 447.4419 447.1240 446.7768
##  [8] 446.3977 445.9841 445.5330 445.0414 444.5059 443.9232 443.2894
## [15] 442.6007 441.8529 441.0421 440.1638 439.2134 438.1866 437.0786
## [22] 435.8849 434.6010 433.2227 431.7459 430.1669 428.4826 426.6902
## [29] 424.7881 422.7751 420.6514 418.4181 416.0778 413.6345 411.0936
## [36] 408.4624 405.7497 402.9656 400.1222 397.2331 394.3130 391.3777
## [43] 388.4440 385.5288 382.6494 379.8225 377.0643 374.3899 371.8128
## [50] 369.3445 366.9950 364.7723 362.6820 360.7275 358.9090 357.2294
## [57] 355.6818 354.2663 352.9767 351.8073 350.7516 349.8028 348.9539
## [64] 348.1962 347.5273 346.9395 346.4260 345.9781 345.5909 345.2624
## [71] 344.9861 344.7585 344.5740 344.4282 344.3170 344.2397 344.1874
## [78] 344.1595 344.1551 344.1657 344.1954 344.2344 344.2845 344.3395
## [85] 344.3993 344.4637 344.5249 344.5859 344.6432 344.6954 344.7438
## [92] 344.7810 344.8133 344.8356 344.8505 344.8545 344.8512 344.8396
## [99] 344.8200
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 344.1551
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.3445
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
##  [1] 449.3181 441.6355 431.8239 423.8310 416.5189 408.8671 400.8204
##  [8] 394.0530 387.9807 382.7897 378.5487 375.0293 371.4923 368.1601
## [15] 365.0829 362.2457 359.3266 356.8310 354.7271 352.9755 351.5117
## [22] 350.2942 349.2838 348.4434 347.7453 347.2320 346.8620 346.5925
## [29] 346.3775 346.2351 346.1350 346.0811 346.0558 346.0634 346.1360
## [36] 346.5438 347.3108 347.9485 348.0291 347.5074 346.4245 345.1235
## [43] 343.9177 342.9170 342.0366 341.3060 340.7598 340.3675 340.1112
## [50] 339.9178 339.9850 340.2275 340.3986 340.4626 340.5583 340.6684
## [57] 340.7713 340.8792 340.9009 340.8598 340.8476 340.9206 341.0049
## [64] 341.1102 341.2112 341.3413 341.4302 341.5151 341.5923 341.6828
## [71] 341.7657
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.9178
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.4923
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
## 1  255.2820965 201886.8  24366.00  226252.8 177520.75     0
## 2  232.6035386 195042.0  24738.86  219780.8 170303.10     1
## 3  211.9396813 186471.9  23875.98  210347.9 162595.89     2
## 4  193.1115442 179632.7  23132.91  202765.6 156499.80     2
## 5  175.9560468 173488.0  22504.97  195993.0 150983.03     3
## 6  160.3245966 167172.3  21744.06  188916.4 145428.26     4
## 7  146.0818013 160657.0  20851.23  181508.2 139805.73     4
## 8  133.1042967 155277.8  20197.07  175474.8 135080.70     4
## 9  121.2796778 150529.0  19683.60  170212.6 130845.43     4
## 10 110.5055255 146528.0  19323.56  165851.5 127204.40     4
## 11 100.6885192 143299.1  19125.24  162424.4 124173.90     5
## 12  91.7436287 140647.0  19086.13  159733.1 121560.88     5
## 13  83.5933775 138006.5  19077.52  157084.0 118928.99     5
## 14  76.1671723 135541.8  19130.13  154672.0 116411.71     5
## 15  69.4006906 133285.5  19245.13  152530.7 114040.40     6
## 16  63.2353245 131222.0  19423.11  150645.1 111798.86     6
## 17  57.6176726 129115.6  19513.87  148629.5 109601.76     6
## 18  52.4990774 127328.3  19636.37  146964.7 107691.96     6
## 19  47.8352040 125831.3  19789.77  145621.1 106041.56     6
## 20  43.5856563 124591.7  19964.55  144556.3 104627.17     6
## 21  39.7136268 123560.5  20149.99  143710.5 103410.47     6
## 22  36.1855776 122706.1  20341.16  143047.2 102364.90     6
## 23  32.9709506 121999.2  20532.05  142531.2 101467.14     6
## 24  30.0419022 121412.8  20719.28  142132.1 100693.50     6
## 25  27.3730624 120926.8  20899.08  141825.9 100027.73     6
## 26  24.9413150 120570.1  21074.42  141644.5  99495.67     6
## 27  22.7255973 120313.3  21236.42  141549.7  99076.85     6
## 28  20.7067179 120126.4  21385.21  141511.6  98741.15     6
## 29  18.8671902 119977.4  21524.97  141502.3  98452.39     6
## 30  17.1910810 119878.8  21661.49  141540.2  98217.28     7
## 31  15.6638727 119809.4  21789.91  141599.3  98019.51     7
## 32  14.2723374 119772.1  21913.18  141685.3  97858.93     7
## 33  13.0044223 119754.6  22029.76  141784.4  97724.85     9
## 34  11.8491453 119759.9  22139.29  141899.2  97620.58     9
## 35  10.7964999 119810.1  22240.74  142050.9  97569.39     9
## 36   9.8373686 120092.6  22323.27  142415.9  97769.31     9
## 37   8.9634439 120624.8  22396.91  143021.7  98227.87     9
## 38   8.1671562 121068.2  22466.96  143535.1  98601.21    11
## 39   7.4416086 121124.3  22445.43  143569.7  98678.83    11
## 40   6.7805166 120761.4  22443.40  143204.8  98317.97    12
## 41   6.1781542 120009.9  22517.74  142527.7  97492.18    12
## 42   5.6293040 119110.2  22609.15  141719.4  96501.05    13
## 43   5.1292121 118279.4  22699.20  140978.6  95580.21    13
## 44   4.6735471 117592.1  22789.36  140381.4  94802.72    13
## 45   4.2583620 116989.1  22873.58  139862.6  94115.49    13
## 46   3.8800609 116489.8  22957.85  139447.6  93531.91    13
## 47   3.5353670 116117.2  23037.19  139154.4  93080.02    13
## 48   3.2212947 115850.0  23109.29  138959.3  92740.72    13
## 49   2.9351238 115675.6  23185.63  138861.2  92489.98    13
## 50   2.6743755 115544.1  23218.87  138763.0  92325.21    13
## 51   2.4367913 115589.8  23262.32  138852.1  92327.47    13
## 52   2.2203135 115754.8  23301.47  139056.2  92453.29    14
## 53   2.0230670 115871.2  23306.57  139177.8  92564.62    15
## 54   1.8433433 115914.8  23327.67  139242.4  92587.09    15
## 55   1.6795857 115979.9  23354.20  139334.1  92625.74    17
## 56   1.5303760 116054.9  23380.18  139435.1  92674.75    17
## 57   1.3944216 116125.1  23404.45  139529.5  92720.65    17
## 58   1.2705450 116198.6  23435.47  139634.1  92763.18    17
## 59   1.1576733 116213.4  23479.07  139692.5  92734.33    17
## 60   1.0548288 116185.4  23535.54  139721.0  92649.87    17
## 61   0.9611207 116177.1  23592.27  139769.3  92584.81    17
## 62   0.8757374 116226.8  23641.87  139868.7  92584.98    17
## 63   0.7979393 116284.4  23676.09  139960.4  92608.26    17
## 64   0.7270526 116356.1  23706.90  140063.0  92649.23    17
## 65   0.6624632 116425.1  23737.80  140162.9  92687.31    18
## 66   0.6036118 116513.9  23763.06  140277.0  92750.85    18
## 67   0.5499886 116574.6  23793.77  140368.4  92780.81    18
## 68   0.5011291 116632.5  23818.12  140450.7  92814.41    17
## 69   0.4566102 116685.3  23843.57  140528.9  92841.71    18
## 70   0.4160462 116747.1  23863.29  140610.4  92883.83    18
## 71   0.3790858 116803.8  23884.79  140688.6  92918.99    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   83.59338
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
