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
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
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
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 588.9958
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 130404.9
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 452.1202 450.6775 449.8964 449.4471 449.1548 448.8355 448.4867
##  [8] 448.1059 447.6904 447.2372 446.7432 446.2051 445.6194 444.9823
## [15] 444.2900 443.5382 442.7228 441.8393 440.8833 439.8500 438.7349
## [22] 437.5332 436.2403 434.8519 433.3638 431.7721 430.0735 428.2651
## [29] 426.3450 424.3119 422.1657 419.9072 417.5389 415.0643 412.4886
## [36] 409.8189 407.0636 404.2326 401.3378 398.3923 395.4109 392.4093
## [43] 389.4039 386.4120 383.4506 380.5368 377.6870 374.9165 372.2395
## [50] 369.6680 367.2123 364.8811 362.6803 360.6145 358.6861 356.8949
## [57] 355.2394 353.7165 352.3219 351.0503 349.8955 348.8509 347.9097
## [64] 347.0641 346.3096 345.6404 345.0468 344.5228 344.0614 343.6595
## [71] 343.3161 343.0198 342.7667 342.5542 342.3765 342.2298 342.1100
## [78] 342.0155 341.9406 341.8816 341.8385 341.8045 341.7810 341.7621
## [85] 341.7474 341.7344 341.7213 341.7077 341.6903 341.6689 341.6416
## [92] 341.6097 341.5724 341.5293 341.4801 341.4259 341.3677 341.3062
## [99] 341.2414
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.2414
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.2123
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
## (Intercept)  154.28043134
## AtBat         -1.94337312
## Hits           6.86658722
## HmRun          1.41607401
## Runs          -1.15621022
## RBI            .         
## Walks          5.67641327
## Years         -7.31403795
## CAtBat        -0.07426903
## CHits          .         
## CHmRun         0.17561659
## CRuns          1.17219652
## CRBI           0.59576865
## CWalks        -0.74065476
## LeagueN       48.38953373
## DivisionW   -116.42613548
## PutOuts        0.28220818
## Assists        0.29718729
## Errors        -2.91727733
## NewLeagueN   -11.76389666
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 207.2074
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
##  [1] 449.7601 439.8135 429.4955 420.8327 412.2487 402.9124 394.1250
##  [8] 386.5485 380.1456 374.9920 370.7983 367.3036 364.2955 361.4940
## [15] 358.6216 355.8416 353.4733 351.4895 349.6943 348.1926 346.9752
## [22] 345.9752 345.1836 344.5442 344.0430 343.6880 343.5498 343.5468
## [29] 343.6300 343.7585 343.8928 344.0646 344.4249 344.9229 345.4122
## [36] 345.8636 346.3585 346.4875 346.3147 345.8392 345.0163 344.1861
## [43] 343.3204 342.6335 342.1348 341.6567 341.0386 340.5539 340.1136
## [50] 339.6912 339.3666 339.1482 339.0766 339.0815 339.0568 339.0259
## [57] 338.7923 338.6091 338.5294 338.5099 338.4884 338.4856 338.5005
## [64] 338.5792 338.6516 338.7251 338.8474 338.9635 339.0787 339.2045
## [71] 339.3152 339.4257 339.5105 339.6148 339.6725 339.7593
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.4856
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 361.494
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
## 1  255.2820965 202284.2  21664.44  223948.6 180619.75     0
## 2  232.6035386 193435.9  21642.37  215078.3 171793.54     1
## 3  211.9396813 184466.4  20942.10  205408.5 163524.30     2
## 4  193.1115442 177100.2  20259.28  197359.4 156840.88     2
## 5  175.9560468 169949.0  19464.46  189413.5 150484.53     3
## 6  160.3245966 162338.4  18840.62  181179.0 143497.80     4
## 7  146.0818013 155334.5  18396.17  173730.7 136938.35     4
## 8  133.1042967 149419.7  18047.25  167467.0 131372.49     4
## 9  121.2796778 144510.7  17788.54  162299.2 126722.13     4
## 10 110.5055255 140619.0  17689.30  158308.3 122929.69     4
## 11 100.6885192 137491.4  17696.39  155187.8 119795.01     5
## 12  91.7436287 134911.9  17763.47  152675.4 117148.47     5
## 13  83.5933775 132711.2  17870.25  150581.5 114840.98     5
## 14  76.1671723 130677.9  18006.40  148684.3 112671.50     5
## 15  69.4006906 128609.5  18219.71  146829.2 110389.75     6
## 16  63.2353245 126623.2  18431.54  145054.8 108191.70     6
## 17  57.6176726 124943.4  18640.94  143584.3 106302.42     6
## 18  52.4990774 123544.9  18816.09  142361.0 104728.81     6
## 19  47.8352040 122286.1  18871.89  141158.0 103414.20     6
## 20  43.5856563 121238.1  18915.42  140153.5 102322.70     6
## 21  39.7136268 120391.8  18969.98  139361.7 101421.79     6
## 22  36.1855776 119698.8  19030.80  138729.6 100668.01     6
## 23  32.9709506 119151.7  19103.11  138254.8 100048.60     6
## 24  30.0419022 118710.7  19176.62  137887.3  99534.07     6
## 25  27.3730624 118365.6  19249.29  137614.9  99116.32     6
## 26  24.9413150 118121.4  19320.77  137442.2  98800.65     6
## 27  22.7255973 118026.4  19384.96  137411.4  98641.48     6
## 28  20.7067179 118024.4  19441.33  137465.7  98583.09     6
## 29  18.8671902 118081.6  19490.96  137572.5  98590.60     6
## 30  17.1910810 118169.9  19536.35  137706.3  98633.56     7
## 31  15.6638727 118262.3  19569.61  137831.9  98692.67     7
## 32  14.2723374 118380.5  19599.70  137980.2  98780.77     7
## 33  13.0044223 118628.5  19622.45  138251.0  99006.08     9
## 34  11.8491453 118971.8  19642.90  138614.7  99328.89     9
## 35  10.7964999 119309.6  19658.51  138968.1  99651.09     9
## 36   9.8373686 119621.6  19674.74  139296.4  99946.88     9
## 37   8.9634439 119964.2  19713.46  139677.7 100250.78     9
## 38   8.1671562 120053.6  19772.46  139826.1 100281.14    11
## 39   7.4416086 119933.8  19818.88  139752.7 100114.97    11
## 40   6.7805166 119604.7  19817.34  139422.1  99787.38    12
## 41   6.1781542 119036.2  19793.98  138830.2  99242.23    12
## 42   5.6293040 118464.1  19730.98  138195.1  98733.08    13
## 43   5.1292121 117868.9  19570.00  137438.9  98298.90    13
## 44   4.6735471 117397.7  19408.32  136806.0  97989.40    13
## 45   4.2583620 117056.2  19284.60  136340.8  97771.65    13
## 46   3.8800609 116729.3  19165.37  135894.7  97563.92    13
## 47   3.5353670 116307.3  18988.21  135295.5  97319.11    13
## 48   3.2212947 115976.9  18818.82  134795.8  97158.11    13
## 49   2.9351238 115677.3  18629.17  134306.4  97048.09    13
## 50   2.6743755 115390.1  18442.69  133832.8  96947.42    13
## 51   2.4367913 115169.7  18264.07  133433.8  96905.63    13
## 52   2.2203135 115021.5  18095.41  133116.9  96926.08    14
## 53   2.0230670 114972.9  17942.38  132915.3  97030.55    15
## 54   1.8433433 114976.2  17832.39  132808.6  97143.86    15
## 55   1.6795857 114959.5  17740.27  132699.8  97219.23    17
## 56   1.5303760 114938.6  17661.72  132600.3  97276.83    17
## 57   1.3944216 114780.2  17584.98  132365.2  97195.22    17
## 58   1.2705450 114656.1  17532.96  132189.1  97123.16    17
## 59   1.1576733 114602.1  17497.86  132100.0  97104.27    17
## 60   1.0548288 114589.0  17466.75  132055.7  97122.20    17
## 61   0.9611207 114574.4  17439.87  132014.2  97134.50    17
## 62   0.8757374 114572.5  17418.58  131991.1  97153.94    17
## 63   0.7979393 114582.6  17405.01  131987.6  97177.59    17
## 64   0.7270526 114635.9  17387.39  132023.3  97248.49    17
## 65   0.6624632 114684.9  17375.90  132060.8  97309.03    18
## 66   0.6036118 114734.7  17368.86  132103.6  97365.86    18
## 67   0.5499886 114817.5  17369.96  132187.5  97447.56    18
## 68   0.5011291 114896.3  17370.51  132266.8  97525.76    17
## 69   0.4566102 114974.4  17375.22  132349.6  97599.14    18
## 70   0.4160462 115059.7  17379.93  132439.6  97679.78    18
## 71   0.3790858 115134.8  17385.98  132520.8  97748.81    18
## 72   0.3454089 115209.8  17390.78  132600.6  97819.03    18
## 73   0.3147237 115267.4  17393.88  132661.2  97873.49    18
## 74   0.2867645 115338.2  17405.35  132743.6  97932.89    18
## 75   0.2612891 115377.4  17399.96  132777.4  97977.47    18
## 76   0.2380769 115436.4  17407.58  132843.9  98028.79    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1  0.8757374   76.16717
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
