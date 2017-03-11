# Shrinkage Methods

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

![](15-shrink_files/figure-latex/unnamed-chunk-7-1.pdf)<!-- --> 

```r
plot(fit_ridge, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/unnamed-chunk-7-2.pdf)<!-- --> 

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

![](15-shrink_files/figure-latex/unnamed-chunk-8-1.pdf)<!-- --> 

The `cv.glmnet()` function returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are smaller than the full linear regression. (As we would expect.)


```r
coef(fit_ridge_cv)
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 295.390327332
## AtBat         0.068388463
## Hits          0.267044986
## HmRun         0.934773737
## Runs          0.438045591
## RBI           0.446352643
## Walks         0.557852274
## Years         1.960352610
## CAtBat        0.005753196
## CHits         0.021745444
## CHmRun        0.162469784
## CRuns         0.043620173
## CRBI          0.045067462
## CWalks        0.045301836
## LeagueN       1.277898191
## DivisionW   -13.838679395
## PutOuts       0.034812580
## Assists       0.005233310
## Errors       -0.067928130
## NewLeagueN    1.404919256
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
## (Intercept) 295.390327332
## AtBat         0.068388463
## Hits          0.267044986
## HmRun         0.934773737
## Runs          0.438045591
## RBI           0.446352643
## Walks         0.557852274
## Years         1.960352610
## CAtBat        0.005753196
## CHits         0.021745444
## CHmRun        0.162469784
## CRuns         0.043620173
## CRBI          0.045067462
## CWalks        0.045301836
## LeagueN       1.277898191
## DivisionW   -13.838679395
## PutOuts       0.034812580
## Assists       0.005233310
## Errors       -0.067928130
## NewLeagueN    1.404919256
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 200.6497
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 145731.4
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 452.8673 450.9427 450.3625 450.0936 449.7996 449.4783 449.1274
##  [8] 448.7443 448.3262 447.8702 447.3733 446.8319 446.2427 445.6018
## [15] 444.9054 444.1491 443.3288 442.4402 441.4786 440.4394 439.3178
## [22] 438.1093 436.8091 435.4130 433.9167 432.3163 430.6086 428.7908
## [29] 426.8609 424.8177 422.6610 420.3920 418.0130 415.5278 412.9418
## [36] 410.2620 407.4972 404.6573 401.7546 398.8024 395.8158 392.8106
## [43] 389.8036 386.8122 383.8540 380.9460 378.1050 375.3464 372.6845
## [50] 370.1314 367.6973 365.3910 363.2192 361.1856 359.2921 357.5389
## [57] 355.9241 354.4444 353.0950 351.8703 350.7637 349.7683 348.8767
## [64] 348.0806 347.3740 346.7552 346.2100 345.7316 345.3135 344.9544
## [71] 344.6515 344.3871 344.1650 343.9822 343.8259 343.6965 343.5929
## [78] 343.5048 343.4337 343.3759 343.3233 343.2798 343.2389 343.2021
## [85] 343.1671 343.1268 343.0849 343.0419 342.9936 342.9412 342.8844
## [92] 342.8238 342.7577 342.6904 342.6140 342.5417 342.4622 342.3866
## [99] 342.3099
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.3099
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 386.8122
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

![](15-shrink_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

```r
plot(fit_lasso, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/unnamed-chunk-10-2.pdf)<!-- --> 

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

![](15-shrink_files/figure-latex/unnamed-chunk-11-1.pdf)<!-- --> 

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
##  [1] 451.6034 445.3248 436.1446 427.9137 420.5790 412.3678 403.9011
##  [8] 396.2950 389.8935 384.5526 380.0834 376.3246 373.1318 370.2864
## [15] 367.6793 365.1228 362.7411 360.6709 358.5417 356.6065 354.9799
## [22] 353.6167 352.5294 351.7168 351.0847 350.5738 350.1778 349.8637
## [29] 349.6410 349.5199 349.4701 349.4641 349.4521 349.6315 350.1294
## [36] 350.8355 351.8152 352.7642 353.0866 352.7220 352.2888 351.2828
## [43] 350.0342 349.1216 348.3497 347.7999 347.3316 346.9260 346.5822
## [50] 346.3504 346.0867 345.8830 345.7815 345.7875 345.9341 346.0362
## [57] 346.1791 346.3604 346.5651 346.7493 346.9799 347.2263 347.3754
## [64] 347.5847 347.6945 347.8225 347.8962 348.0228 348.1478 348.2099
## [71] 348.2789
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 345.7815
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 376.3246
```

## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
#lassoModCV
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high  conf.low nzero
## 1  255.2820965 203945.6  37790.57  241736.2 166155.07     0
## 2  232.6035386 198314.2  38370.94  236685.1 159943.26     1
## 3  211.9396813 190222.1  37418.73  227640.8 152803.35     2
## 4  193.1115442 183110.1  35977.22  219087.3 147132.89     2
## 5  175.9560468 176886.7  34805.75  211692.5 142080.97     3
## 6  160.3245966 170047.2  33819.32  203866.5 136227.84     4
## 7  146.0818013 163136.1  32665.30  195801.4 130470.81     4
## 8  133.1042967 157049.7  31573.65  188623.4 125476.05     4
## 9  121.2796778 152016.9  30655.62  182672.5 121361.30     4
## 10 110.5055255 147880.7  29899.85  177780.6 117980.87     4
## 11 100.6885192 144463.4  29293.17  173756.6 115170.26     5
## 12  91.7436287 141620.2  28800.72  170420.9 112819.47     5
## 13  83.5933775 139227.3  28363.68  167591.0 110863.63     5
## 14  76.1671723 137112.0  27879.02  164991.0 109232.97     5
## 15  69.4006906 135188.0  27365.50  162553.5 107822.55     6
## 16  63.2353245 133314.7  26894.57  160209.2 106420.10     6
## 17  57.6176726 131581.1  26496.04  158077.1 105085.07     6
## 18  52.4990774 130083.5  26153.45  156236.9 103930.04     6
## 19  47.8352040 128552.2  25695.51  154247.7 102856.66     6
## 20  43.5856563 127168.2  25256.99  152425.2 101911.21     6
## 21  39.7136268 126010.7  24895.04  150905.8 101115.67     6
## 22  36.1855776 125044.8  24596.11  149640.9 100448.66     6
## 23  32.9709506 124277.0  24345.73  148622.7  99931.27     6
## 24  30.0419022 123704.7  24130.30  147835.0  99574.42     6
## 25  27.3730624 123260.5  23950.26  147210.7  99310.21     6
## 26  24.9413150 122902.0  23804.77  146706.8  99097.22     6
## 27  22.7255973 122624.5  23682.73  146307.2  98941.77     6
## 28  20.7067179 122404.6  23582.56  145987.2  98822.04     6
## 29  18.8671902 122248.8  23499.45  145748.3  98749.39     6
## 30  17.1910810 122164.2  23423.44  145587.6  98740.75     7
## 31  15.6638727 122129.3  23357.73  145487.1  98771.60     7
## 32  14.2723374 122125.2  23305.80  145431.0  98819.38     7
## 33  13.0044223 122116.8  23260.81  145377.6  98855.98     9
## 34  11.8491453 122242.2  23216.99  145459.2  99025.22     9
## 35  10.7964999 122590.6  23249.63  145840.2  99340.97     9
## 36   9.8373686 123085.6  23328.54  146414.1  99757.04     9
## 37   8.9634439 123774.0  23402.69  147176.7 100371.28     9
## 38   8.1671562 124442.6  23501.37  147944.0 100941.24    11
## 39   7.4416086 124670.1  23510.48  148180.6 101159.65    11
## 40   6.7805166 124412.8  23391.03  147803.9 101021.79    12
## 41   6.1781542 124107.4  23238.14  147345.6 100869.28    12
## 42   5.6293040 123399.6  23108.81  146508.4 100290.78    13
## 43   5.1292121 122523.9  23038.32  145562.3  99485.62    13
## 44   4.6735471 121885.9  22969.47  144855.4  98916.44    13
## 45   4.2583620 121347.5  22933.80  144281.3  98413.71    13
## 46   3.8800609 120964.8  22898.03  143862.8  98066.76    13
## 47   3.5353670 120639.2  22860.95  143500.2  97778.26    13
## 48   3.2212947 120357.6  22819.45  143177.1  97538.17    13
## 49   2.9351238 120119.2  22780.13  142899.4  97339.10    13
## 50   2.6743755 119958.6  22738.96  142697.6  97219.66    13
## 51   2.4367913 119776.0  22695.04  142471.1  97080.97    13
## 52   2.2203135 119635.1  22645.54  142280.6  96989.52    14
## 53   2.0230670 119564.8  22611.10  142175.9  96953.73    15
## 54   1.8433433 119569.0  22585.34  142154.3  96983.66    15
## 55   1.6795857 119670.4  22575.39  142245.8  97094.99    17
## 56   1.5303760 119741.0  22516.22  142257.2  97224.82    17
## 57   1.3944216 119840.0  22462.11  142302.1  97377.85    17
## 58   1.2705450 119965.5  22417.97  142383.5  97547.54    17
## 59   1.1576733 120107.4  22379.34  142486.7  97728.02    17
## 60   1.0548288 120235.1  22346.63  142581.7  97888.45    17
## 61   0.9611207 120395.0  22314.79  142709.8  98080.25    17
## 62   0.8757374 120566.1  22320.41  142886.5  98245.70    17
## 63   0.7979393 120669.6  22269.84  142939.5  98399.81    17
## 64   0.7270526 120815.1  22268.60  143083.7  98546.53    17
## 65   0.6624632 120891.5  22226.08  143117.6  98665.40    18
## 66   0.6036118 120980.5  22217.11  143197.6  98763.39    18
## 67   0.5499886 121031.8  22184.73  143216.5  98847.02    18
## 68   0.5011291 121119.9  22178.23  143298.1  98941.66    17
## 69   0.4566102 121206.9  22184.26  143391.2  99022.66    18
## 70   0.4160462 121250.1  22141.72  143391.8  99108.40    18
## 71   0.3790858 121298.2  22133.34  143431.5  99164.85    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.023067   91.74363
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

![](15-shrink_files/figure-latex/unnamed-chunk-16-1.pdf)<!-- --> 


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

![](15-shrink_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 


```r
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```

![](15-shrink_files/figure-latex/unnamed-chunk-21-1.pdf)<!-- --> 

We can extracting the two relevant $\lambda$ values.


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
fit_lasso = train(
  x = X,
  y = y,
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

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.3.2 and the following packages:

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
##  [1] "Rcpp"         "compiler"     "nloptr"       "plyr"        
##  [5] "class"        "iterators"    "tools"        "lme4"        
##  [9] "digest"       "evaluate"     "tibble"       "nlme"        
## [13] "gtable"       "mgcv"         "psych"        "DBI"         
## [17] "yaml"         "parallel"     "SparseM"      "e1071"       
## [21] "dplyr"        "stringr"      "knitr"        "MatrixModels"
## [25] "stats4"       "nnet"         "rprojroot"    "grid"        
## [29] "R6"           "foreign"      "rmarkdown"    "bookdown"    
## [33] "minqa"        "car"          "reshape2"     "tidyr"       
## [37] "magrittr"     "splines"      "MASS"         "ModelMetrics"
## [41] "backports"    "scales"       "codetools"    "htmltools"   
## [45] "pbkrtest"     "assertthat"   "mnormt"       "colorspace"  
## [49] "quantreg"     "stringi"      "lazyeval"     "munsell"
```
