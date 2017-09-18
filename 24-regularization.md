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
##  [1] 452.6564 450.9968 450.2320 449.9641 449.6713 449.3513 449.0018
##  [8] 448.6202 448.2037 447.7496 447.2545 446.7153 446.1283 445.4899
## [15] 444.7961 444.0426 443.2254 442.3401 441.3820 440.3464 439.2287
## [22] 438.0243 436.7285 435.3368 433.8451 432.2496 430.5467 428.7338
## [29] 426.8087 424.7701 422.6178 420.3528 417.9772 415.4947 412.9103
## [36] 410.2310 407.4651 404.6225 401.7148 398.7552 395.7582 392.7393
## [43] 389.7150 386.7022 383.7180 380.7792 377.9022 375.1022 372.3930
## [50] 369.7869 367.2940 364.9230 362.6803 360.5702 358.5947 356.7542
## [57] 355.0472 353.4707 352.0206 350.6915 349.4776 348.3723 347.3689
## [64] 346.4600 345.6406 344.9057 344.2457 343.6540 343.1252 342.6566
## [71] 342.2410 341.8742 341.5511 341.2692 341.0185 340.8016 340.6144
## [78] 340.4490 340.3070 340.1817 340.0738 339.9765 339.8922 339.8148
## [85] 339.7439 339.6787 339.6169 339.5567 339.4980 339.4391 339.3799
## [92] 339.3197 339.2586 339.1954 339.1325 339.0672 339.0026 338.9386
## [99] 338.8744
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.8744
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 364.923
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
##                        1
## (Intercept)  117.5258439
## AtBat         -1.4742901
## Hits           5.4994256
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.5991651
## Years         -9.1918308
## CAtBat         .        
## CHits          .        
## CHmRun         0.4806743
## CRuns          0.6354799
## CRBI           0.3956153
## CWalks        -0.4993240
## LeagueN       31.6238174
## DivisionW   -119.2516409
## PutOuts        0.2704287
## Assists        0.1594997
## Errors        -1.9426357
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 176.0238
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
##  [1] 450.3228 441.4492 431.2887 422.7981 415.1785 407.2484 398.7369
##  [8] 390.9078 384.2152 378.4727 373.4599 369.0101 364.9040 361.2576
## [15] 357.7893 354.4198 351.4229 348.8458 346.6937 344.8989 343.4015
## [22] 342.1545 341.1174 340.2701 339.5857 339.0694 338.7137 338.4680
## [29] 338.3028 338.2077 338.1431 338.1058 338.0506 338.0746 338.3434
## [36] 338.9465 339.8521 340.7859 341.4840 341.6905 341.0559 339.9710
## [43] 338.8451 337.9531 337.1877 336.5962 336.1522 335.8261 335.6668
## [50] 335.8431 336.3625 336.9754 337.6084 338.2207 338.7925 339.3071
## [57] 339.6849 340.0470 340.3752 340.6976 341.0050 341.3162 341.5322
## [64] 341.7544 341.8459 342.0429 342.1688 342.3308 342.4303 342.5076
## [71] 342.6712 342.7563 342.8310 342.9097 343.0038
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 335.6668
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 361.2576
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
## 1  255.2820965 202790.6  34117.91  236908.6 168672.73     0
## 2  232.6035386 194877.4  33953.52  228830.9 160923.87     1
## 3  211.9396813 186009.9  32741.96  218751.9 153267.95     2
## 4  193.1115442 178758.2  31732.58  210490.8 147025.62     2
## 5  175.9560468 172373.2  30954.40  203327.6 141418.82     3
## 6  160.3245966 165851.2  30205.99  196057.2 135645.25     4
## 7  146.0818013 158991.1  29270.89  188262.0 129720.23     4
## 8  133.1042967 152808.9  28394.34  181203.2 124414.55     4
## 9  121.2796778 147621.3  27629.80  175251.1 119991.52     4
## 10 110.5055255 143241.6  26961.62  170203.2 116279.98     4
## 11 100.6885192 139472.3  26351.58  165823.9 113120.69     5
## 12  91.7436287 136168.4  25784.83  161953.3 110383.59     5
## 13  83.5933775 133155.0  25297.80  158452.8 107857.17     5
## 14  76.1671723 130507.1  24896.21  155403.3 105610.85     5
## 15  69.4006906 128013.2  24563.18  152576.3 103449.98     6
## 16  63.2353245 125613.4  24292.86  149906.3 101320.57     6
## 17  57.6176726 123498.0  24005.59  147503.6  99492.45     6
## 18  52.4990774 121693.4  23717.71  145411.1  97975.67     6
## 19  47.8352040 120196.5  23469.26  143665.8  96727.24     6
## 20  43.5856563 118955.3  23255.06  142210.3  95700.20     6
## 21  39.7136268 117924.6  23069.87  140994.4  94854.70     6
## 22  36.1855776 117069.7  22909.91  139979.6  94159.80     6
## 23  32.9709506 116361.1  22771.50  139132.6  93589.56     6
## 24  30.0419022 115783.7  22649.21  138432.9  93134.52     6
## 25  27.3730624 115318.4  22546.47  137864.9  92771.97     6
## 26  24.9413150 114968.0  22466.67  137434.7  92501.36     6
## 27  22.7255973 114727.0  22413.73  137140.7  92313.24     6
## 28  20.7067179 114560.6  22370.76  136931.3  92189.81     6
## 29  18.8671902 114448.8  22333.43  136782.2  92115.36     6
## 30  17.1910810 114384.5  22301.84  136686.3  92082.64     7
## 31  15.6638727 114340.8  22273.29  136614.1  92067.48     7
## 32  14.2723374 114315.5  22248.00  136563.5  92067.54     7
## 33  13.0044223 114278.2  22222.18  136500.4  92056.01     9
## 34  11.8491453 114294.4  22199.26  136493.7  92095.16     9
## 35  10.7964999 114476.3  22177.09  136653.4  92299.19     9
## 36   9.8373686 114884.7  22119.98  137004.7  92764.73     9
## 37   8.9634439 115499.4  22027.91  137527.3  93471.51     9
## 38   8.1671562 116135.0  21937.82  138072.9  94197.22    11
## 39   7.4416086 116611.3  21810.50  138421.8  94800.83    11
## 40   6.7805166 116752.4  21595.66  138348.0  95156.71    12
## 41   6.1781542 116319.1  21244.35  137563.5  95074.79    12
## 42   5.6293040 115580.3  20859.66  136439.9  94720.61    13
## 43   5.1292121 114816.0  20538.64  135354.7  94277.39    13
## 44   4.6735471 114212.3  20247.76  134460.0  93964.54    13
## 45   4.2583620 113695.5  19980.52  133676.1  93715.00    13
## 46   3.8800609 113297.0  19739.66  133036.6  93557.31    13
## 47   3.5353670 112998.3  19523.78  132522.1  93474.52    13
## 48   3.2212947 112779.2  19322.80  132102.0  93456.39    13
## 49   2.9351238 112672.2  19146.40  131818.6  93525.79    13
## 50   2.6743755 112790.6  19021.33  131811.9  93769.28    13
## 51   2.4367913 113139.8  18934.04  132073.8  94205.73    13
## 52   2.2203135 113552.4  18862.02  132414.4  94690.40    14
## 53   2.0230670 113979.5  18805.62  132785.1  95173.83    15
## 54   1.8433433 114393.2  18760.90  133154.1  95632.32    15
## 55   1.6795857 114780.3  18724.07  133504.4  96056.27    17
## 56   1.5303760 115129.3  18690.37  133819.7  96438.97    17
## 57   1.3944216 115385.9  18688.56  134074.4  96697.30    17
## 58   1.2705450 115632.0  18701.15  134333.1  96930.84    17
## 59   1.1576733 115855.2  18722.38  134577.6  97132.87    17
## 60   1.0548288 116074.9  18746.75  134821.6  97328.11    17
## 61   0.9611207 116284.4  18769.50  135053.9  97514.92    17
## 62   0.8757374 116496.8  18792.14  135288.9  97704.64    17
## 63   0.7979393 116644.3  18814.21  135458.5  97830.05    17
## 64   0.7270526 116796.1  18831.21  135627.3  97964.88    17
## 65   0.6624632 116858.6  18850.01  135708.6  98008.58    18
## 66   0.6036118 116993.3  18868.66  135862.0  98124.67    18
## 67   0.5499886 117079.5  18892.24  135971.7  98187.26    18
## 68   0.5011291 117190.4  18908.15  136098.5  98282.22    17
## 69   0.4566102 117258.5  18919.10  136177.6  98339.40    18
## 70   0.4160462 117311.5  18934.04  136245.5  98377.44    18
## 71   0.3790858 117423.5  18946.78  136370.3  98476.76    18
## 72   0.3454089 117481.8  18965.71  136447.6  98516.14    18
## 73   0.3147237 117533.1  18975.50  136508.6  98557.60    18
## 74   0.2867645 117587.1  18990.30  136577.4  98596.80    18
## 75   0.2612891 117651.6  18998.45  136650.1  98653.16    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.935124   76.16717
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
