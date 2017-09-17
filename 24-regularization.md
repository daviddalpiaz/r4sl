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
## (Intercept) 159.796625075
## AtBat         0.102483884
## Hits          0.446840519
## HmRun         1.289060569
## Runs          0.702915318
## RBI           0.686866069
## Walks         0.925962429
## Years         2.714623469
## CAtBat        0.008746278
## CHits         0.034359576
## CHmRun        0.253594871
## CRuns         0.068874010
## CRBI          0.071334608
## CWalks        0.066114944
## LeagueN       5.396487460
## DivisionW   -29.096663826
## PutOuts       0.067805863
## Assists       0.009201998
## Errors       -0.235989099
## NewLeagueN    4.457548079
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
## (Intercept) 159.796625075
## AtBat         0.102483884
## Hits          0.446840519
## HmRun         1.289060569
## Runs          0.702915318
## RBI           0.686866069
## Walks         0.925962429
## Years         2.714623469
## CAtBat        0.008746278
## CHits         0.034359576
## CHmRun        0.253594871
## CRuns         0.068874010
## CRBI          0.071334608
## CWalks        0.066114944
## LeagueN       5.396487460
## DivisionW   -29.096663826
## PutOuts       0.067805863
## Assists       0.009201998
## Errors       -0.235989099
## NewLeagueN    4.457548079
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 906.8121
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 125141.2
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.7940 450.2219 449.3478 449.0816 448.7906 448.4727 448.1253
##  [8] 447.7461 447.3323 446.8811 446.3892 445.8533 445.2701 444.6357
## [15] 443.9463 443.1977 442.3858 441.5061 440.5541 439.5253 438.4149
## [22] 437.2183 435.9309 434.5484 433.0665 431.4815 429.7899 427.9890
## [29] 426.0767 424.0517 421.9138 419.6638 417.3040 414.8380 412.2707
## [36] 409.6088 406.8609 404.0365 401.1470 398.2055 395.2264 392.2248
## [43] 389.2170 386.2197 383.2496 380.3235 377.4573 374.6660 371.9633
## [50] 369.3613 366.8699 364.4979 362.2517 360.1352 358.1508 356.2987
## [57] 354.5776 352.9848 351.5160 350.1663 348.9298 347.8001 346.7707
## [64] 345.8347 344.9890 344.2233 343.5310 342.9069 342.3451 341.8437
## [71] 341.3941 340.9882 340.6242 340.3030 340.0131 339.7511 339.5161
## [78] 339.3027 339.1071 338.9263 338.7584 338.5978 338.4450 338.2960
## [85] 338.1488 338.0013 337.8515 337.6983 337.5407 337.3771 337.2079
## [92] 337.0322 336.8494 336.6598 336.4637 336.2619 336.0550 335.8436
## [99] 335.6292
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 335.6292
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 360.1352
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
##  [1] 450.9849 441.2663 430.7131 421.8783 413.7090 405.1249 396.5367
##  [8] 388.8678 382.1683 376.5896 371.9840 368.0131 364.5977 361.6205
## [15] 358.9219 356.0451 353.0435 350.4744 348.3306 346.5454 345.0615
## [22] 343.8281 342.8128 341.9771 341.3509 340.9237 340.6474 340.4549
## [29] 340.3745 340.3513 340.3874 340.4263 340.4585 340.5022 340.5521
## [36] 340.7633 341.0115 341.2599 341.1478 340.6671 339.8857 338.9807
## [43] 338.0488 337.1893 336.5260 335.9726 335.4815 335.1694 334.9217
## [50] 334.8595 334.8460 334.8496 335.0051 335.2395 335.5023 335.7140
## [57] 335.8809 335.9289 335.9748 336.0922 336.2372 336.3936 336.5661
## [64] 336.7375 336.9261 337.0936 337.2845 337.4877 337.6629 337.8375
## [71] 337.9856 338.1346 338.2591
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 334.846
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 361.6205
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
## 1  255.2820965 203387.3  26437.80  229825.1 176949.55     0
## 2  232.6035386 194715.9  25892.42  220608.4 168823.51     1
## 3  211.9396813 185513.8  25048.14  210561.9 160465.64     2
## 4  193.1115442 177981.3  24386.31  202367.6 153594.97     2
## 5  175.9560468 171155.1  23909.74  195064.8 147245.36     3
## 6  160.3245966 164126.2  23704.60  187830.8 140421.62     4
## 7  146.0818013 157241.3  23458.33  180699.7 133783.00     4
## 8  133.1042967 151218.2  23228.28  174446.5 127989.89     4
## 9  121.2796778 146052.6  23047.40  169100.0 123005.18     4
## 10 110.5055255 141819.8  22939.52  164759.3 118880.23     4
## 11 100.6885192 138372.1  22887.81  161259.9 115484.28     5
## 12  91.7436287 135433.7  22897.08  158330.7 112536.57     5
## 13  83.5933775 132931.5  22916.92  155848.4 110014.54     5
## 14  76.1671723 130769.4  22959.99  153729.4 107809.42     5
## 15  69.4006906 128825.0  23039.55  151864.5 105785.41     6
## 16  63.2353245 126768.1  23031.50  149799.6 103736.61     6
## 17  57.6176726 124639.7  22861.34  147501.0 101778.36     6
## 18  52.4990774 122832.3  22706.74  145539.1 100125.58     6
## 19  47.8352040 121334.2  22575.15  143909.3  98759.04     6
## 20  43.5856563 120093.7  22462.86  142556.6  97630.85     6
## 21  39.7136268 119067.4  22367.24  141434.7  96700.18     6
## 22  36.1855776 118217.8  22285.29  140503.1  95932.49     6
## 23  32.9709506 117520.6  22215.99  139736.6  95304.65     6
## 24  30.0419022 116948.3  22157.66  139106.0  94790.65     6
## 25  27.3730624 116520.4  22105.43  138625.9  94414.99     6
## 26  24.9413150 116229.0  22049.73  138278.7  94179.27     6
## 27  22.7255973 116040.6  21996.90  138037.5  94043.75     6
## 28  20.7067179 115909.6  21953.81  137863.4  93955.74     6
## 29  18.8671902 115854.8  21915.76  137770.6  93939.06     6
## 30  17.1910810 115839.0  21881.33  137720.3  93957.69     7
## 31  15.6638727 115863.6  21848.80  137712.4  94014.79     7
## 32  14.2723374 115890.1  21825.52  137715.6  94064.53     7
## 33  13.0044223 115912.0  21805.78  137717.8  94106.23     9
## 34  11.8491453 115941.8  21788.44  137730.2  94153.33     9
## 35  10.7964999 115975.7  21773.60  137749.3  94202.12     9
## 36   9.8373686 116119.7  21752.89  137872.5  94366.77     9
## 37   8.9634439 116288.9  21731.17  138020.0  94557.69     9
## 38   8.1671562 116458.3  21710.78  138169.1  94747.54    11
## 39   7.4416086 116381.8  21657.06  138038.9  94724.77    11
## 40   6.7805166 116054.1  21412.37  137466.5  94641.72    12
## 41   6.1781542 115522.3  21180.04  136702.4  94342.28    12
## 42   5.6293040 114907.9  20969.25  135877.2  93938.68    13
## 43   5.1292121 114277.0  20788.75  135065.7  93488.24    13
## 44   4.6735471 113696.6  20647.58  134344.2  93049.01    13
## 45   4.2583620 113249.8  20548.60  133798.4  92701.16    13
## 46   3.8800609 112877.6  20461.09  133338.7  92416.52    13
## 47   3.5353670 112547.8  20375.44  132923.3  92172.39    13
## 48   3.2212947 112338.6  20331.59  132670.1  92006.97    13
## 49   2.9351238 112172.5  20297.23  132469.7  91875.30    13
## 50   2.6743755 112130.9  20286.90  132417.8  91843.99    13
## 51   2.4367913 112121.8  20272.15  132394.0  91849.66    13
## 52   2.2203135 112124.2  20256.74  132381.0  91867.50    14
## 53   2.0230670 112228.4  20259.63  132488.0  91968.76    15
## 54   1.8433433 112385.6  20284.41  132670.0  92101.14    15
## 55   1.6795857 112561.8  20306.62  132868.4  92255.15    17
## 56   1.5303760 112703.9  20310.74  133014.7  92393.17    17
## 57   1.3944216 112816.0  20342.68  133158.7  92473.29    17
## 58   1.2705450 112848.3  20387.37  133235.6  92460.88    17
## 59   1.1576733 112879.1  20431.05  133310.1  92448.01    17
## 60   1.0548288 112957.9  20468.03  133426.0  92489.92    17
## 61   0.9611207 113055.4  20498.38  133553.8  92557.06    17
## 62   0.8757374 113160.6  20524.32  133685.0  92636.33    17
## 63   0.7979393 113276.8  20548.70  133825.5  92728.07    17
## 64   0.7270526 113392.1  20574.02  133966.1  92818.12    17
## 65   0.6624632 113519.2  20597.60  134116.8  92921.58    18
## 66   0.6036118 113632.1  20626.17  134258.3  93005.95    18
## 67   0.5499886 113760.8  20656.13  134416.9  93104.69    18
## 68   0.5011291 113898.0  20690.52  134588.5  93207.43    17
## 69   0.4566102 114016.3  20721.27  134737.5  93295.00    18
## 70   0.4160462 114134.1  20759.95  134894.1  93374.20    18
## 71   0.3790858 114234.2  20789.75  135024.0  93444.50    18
## 72   0.3454089 114335.0  20822.27  135157.2  93512.71    18
## 73   0.3147237 114419.2  20847.81  135267.0  93571.40    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   76.16717
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
