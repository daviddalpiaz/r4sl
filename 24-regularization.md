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
##  [1] 452.0947 450.2515 449.6132 449.3408 449.0432 448.7179 448.3626
##  [8] 447.9747 447.5514 447.0897 446.5865 446.0384 445.4417 444.7928
## [15] 444.0876 443.3218 442.4912 441.5914 440.6176 439.5652 438.4294
## [22] 437.2054 435.8887 434.4747 432.9591 431.3382 429.6084 427.7669
## [29] 425.8118 423.7417 421.5565 419.2573 416.8463 414.3274 411.7060
## [36] 408.9890 406.1853 403.3049 400.3600 397.3641 394.3322 391.2803
## [43] 388.2253 385.1846 382.1758 379.2162 376.3224 373.5101 370.7934
## [50] 368.1846 365.6940 363.3301 361.0997 359.0065 357.0525 355.2376
## [57] 353.5600 352.0162 350.6015 349.3101 348.1356 347.0710 346.1091
## [64] 345.2418 344.4636 343.7717 343.1506 342.5971 342.1040 341.6716
## [71] 341.2925 340.9570 340.6601 340.4029 340.1808 339.9810 339.8090
## [78] 339.6594 339.5256 339.4086 339.3002 339.2062 339.1129 339.0283
## [85] 338.9471 338.8632 338.7808 338.6959 338.6087 338.5190 338.4244
## [92] 338.3240 338.2230 338.1151 338.0069 337.8943 337.7804 337.6661
## [99] 337.5512
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.5512
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 368.1846
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
## (Intercept) 193.74263858
## AtBat         .         
## Hits          1.21471320
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.28957902
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.12923755
## CRBI          0.31515925
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.02533305
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
## (Intercept) 193.74263858
## AtBat         .         
## Hits          1.21471320
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.28957902
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.12923755
## CRBI          0.31515925
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.02533305
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 2.974022
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 127112.4
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 450.1954 441.3498 430.8763 421.7439 412.9631 403.4383 394.6617
##  [8] 387.0535 380.5320 375.1012 370.6620 366.9032 363.5072 360.4005
## [15] 357.3590 354.5919 352.2055 349.9386 348.0385 346.4742 345.1863
## [22] 344.1295 343.2601 342.5573 341.9791 341.5198 341.1618 340.9046
## [29] 340.7464 340.6382 340.5454 340.4940 340.4495 340.4264 340.3968
## [36] 340.4323 340.7440 340.9221 340.9708 340.8004 340.6725 340.6202
## [43] 339.8653 338.8559 337.9899 337.2740 336.7009 336.2644 335.9521
## [50] 335.7499 335.7564 335.7701 335.7879 335.7761 335.7445 335.7720
## [57] 335.8457 335.8739 335.9450 336.0834 336.2578 336.4489 336.6582
## [64] 336.7371 336.9648 337.0198 337.2152 337.2837 337.4342 337.5588
## [71] 337.5868 337.8380 337.8298
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 335.7445
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 366.9032
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
## 1  255.2820965 202675.9  39364.99  242040.8 163310.88     0
## 2  232.6035386 194789.6  39673.66  234463.3 155115.97     1
## 3  211.9396813 185654.4  38366.48  224020.9 147287.91     2
## 4  193.1115442 177867.9  37109.82  214977.7 140758.07     2
## 5  175.9560468 170538.5  35998.58  206537.1 134539.95     3
## 6  160.3245966 162762.4  34916.90  197679.3 127845.53     4
## 7  146.0818013 155757.9  33893.88  189651.7 121863.99     4
## 8  133.1042967 149810.4  32972.98  182783.4 116837.47     4
## 9  121.2796778 144804.6  32191.13  176995.7 112613.47     4
## 10 110.5055255 140700.9  31538.60  172239.5 109162.33     4
## 11 100.6885192 137390.3  31014.53  168404.9 106375.81     5
## 12  91.7436287 134617.9  30549.00  165166.9 104068.93     5
## 13  83.5933775 132137.5  30085.56  162223.1 102051.93     5
## 14  76.1671723 129888.5  29692.74  159581.2 100195.75     5
## 15  69.4006906 127705.5  29375.08  157080.6  98330.40     6
## 16  63.2353245 125735.4  29115.46  154850.9  96619.99     6
## 17  57.6176726 124048.7  28895.84  152944.6  95152.88     6
## 18  52.4990774 122457.0  28556.79  151013.8  93900.26     6
## 19  47.8352040 121130.8  28237.40  149368.2  92893.41     6
## 20  43.5856563 120044.4  27952.97  147997.4  92091.41     6
## 21  39.7136268 119153.6  27698.89  146852.4  91454.67     6
## 22  36.1855776 118425.1  27471.84  145896.9  90953.26     6
## 23  32.9709506 117827.5  27267.79  145095.3  90559.72     6
## 24  30.0419022 117345.5  27084.57  144430.1  90260.96     6
## 25  27.3730624 116949.7  26919.79  143869.5  90029.89     6
## 26  24.9413150 116635.8  26770.35  143406.1  89865.45     6
## 27  22.7255973 116391.3  26634.63  143026.0  89756.72     6
## 28  20.7067179 116215.9  26517.92  142733.8  89697.99     6
## 29  18.8671902 116108.1  26434.82  142542.9  89673.29     6
## 30  17.1910810 116034.4  26360.87  142395.3  89673.53     7
## 31  15.6638727 115971.2  26298.98  142270.1  89672.18     7
## 32  14.2723374 115936.2  26254.96  142191.1  89681.21     7
## 33  13.0044223 115905.9  26209.02  142114.9  89696.87     9
## 34  11.8491453 115890.1  26170.16  142060.3  89719.96     9
## 35  10.7964999 115870.0  26142.53  142012.5  89727.44     9
## 36   9.8373686 115894.1  26113.88  142008.0  89780.24     9
## 37   8.9634439 116106.5  26092.13  142198.6  90014.34     9
## 38   8.1671562 116227.9  26081.66  142309.6  90146.24    11
## 39   7.4416086 116261.1  26081.56  142342.6  90179.52    11
## 40   6.7805166 116144.9  26046.87  142191.8  90098.04    12
## 41   6.1781542 116057.8  26005.87  142063.6  90051.88    12
## 42   5.6293040 116022.1  25950.18  141972.3  90071.91    13
## 43   5.1292121 115508.4  25593.86  141102.3  89914.57    13
## 44   4.6735471 114823.3  25193.94  140017.3  89629.40    13
## 45   4.2583620 114237.2  24833.47  139070.6  89403.68    13
## 46   3.8800609 113753.8  24508.69  138262.5  89245.08    13
## 47   3.5353670 113367.5  24216.68  137584.2  89150.85    13
## 48   3.2212947 113073.8  23951.89  137025.7  89121.87    13
## 49   2.9351238 112863.8  23714.28  136578.1  89149.51    13
## 50   2.6743755 112728.0  23480.65  136208.6  89247.32    13
## 51   2.4367913 112732.4  23205.30  135937.7  89527.05    13
## 52   2.2203135 112741.6  22940.26  135681.8  89801.29    14
## 53   2.0230670 112753.5  22714.19  135467.7  90039.33    15
## 54   1.8433433 112745.6  22518.51  135264.1  90227.10    15
## 55   1.6795857 112724.3  22342.42  135066.8  90381.92    17
## 56   1.5303760 112742.8  22193.07  134935.9  90549.76    17
## 57   1.3944216 112792.4  22084.29  134876.7  90708.08    17
## 58   1.2705450 112811.3  21983.46  134794.7  90827.81    17
## 59   1.1576733 112859.1  21906.72  134765.8  90952.35    17
## 60   1.0548288 112952.0  21839.52  134791.6  91112.53    17
## 61   0.9611207 113069.3  21779.64  134849.0  91289.68    17
## 62   0.8757374 113197.8  21729.28  134927.1  91468.55    17
## 63   0.7979393 113338.7  21682.91  135021.6  91655.82    17
## 64   0.7270526 113391.8  21604.14  134996.0  91787.70    17
## 65   0.6624632 113545.3  21515.73  135061.0  92029.53    18
## 66   0.6036118 113582.3  21432.92  135015.3  92149.41    18
## 67   0.5499886 113714.1  21367.38  135081.5  92346.69    18
## 68   0.5011291 113760.3  21308.71  135069.0  92451.61    17
## 69   0.4566102 113861.9  21247.66  135109.5  92614.21    18
## 70   0.4160462 113945.9  21240.40  135186.3  92705.54    18
## 71   0.3790858 113964.9  21164.65  135129.5  92800.22    18
## 72   0.3454089 114134.5  21167.41  135301.9  92967.10    18
## 73   0.3147237 114129.0  21102.76  135231.7  93026.21    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   1.679586   91.74363
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
