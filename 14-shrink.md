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

![](14-shrink_files/figure-latex/ridge-1.pdf)<!-- --> 

```r
plot(fit_ridge, xvar = "lambda", label = TRUE)
```

![](14-shrink_files/figure-latex/ridge-2.pdf)<!-- --> 

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

![](14-shrink_files/figure-latex/unnamed-chunk-8-1.pdf)<!-- --> 

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
##  [1] 450.5924 449.1330 448.2346 447.9659 447.6721 447.3511 447.0005
##  [8] 446.6177 446.2000 445.7444 445.2478 444.7069 444.1182 443.4778
## [15] 442.7818 442.0261 441.2065 440.3185 439.3575 438.3190 437.1981
## [22] 435.9902 434.6908 433.2954 431.7998 430.2001 428.4930 426.6757
## [29] 424.7461 422.7031 420.5464 418.2772 415.8975 413.4113 410.8237
## [36] 408.1417 405.3739 402.5303 399.6228 396.6647 393.6708 390.6569
## [43] 387.6396 384.6362 381.6639 378.7398 375.8805 373.1013 370.4165
## [50] 367.8382 365.3764 363.0399 360.8356 358.7672 356.8370 355.0449
## [57] 353.3894 351.8673 350.4741 349.2045 348.0521 347.0103 346.0720
## [64] 345.2291 344.4774 343.8115 343.2212 342.6992 342.2391 341.8396
## [71] 341.4948 341.1955 340.9419 340.7242 340.5389 340.3855 340.2584
## [78] 340.1510 340.0611 339.9894 339.9252 339.8700 339.8213 339.7754
## [85] 339.7307 339.6822 339.6320 339.5781 339.5142 339.4432 339.3640
## [92] 339.2730 339.1756 339.0652 338.9458 338.8179 338.6779 338.5325
## [99] 338.3777
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.3777
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 365.3764
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

![](14-shrink_files/figure-latex/lasso-1.pdf)<!-- --> 

```r
plot(fit_lasso, xvar = "lambda", label = TRUE)
```

![](14-shrink_files/figure-latex/lasso-2.pdf)<!-- --> 

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

![](14-shrink_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

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
##  [1] 451.7611 444.2250 434.0957 425.2656 417.0067 407.9086 399.5300
##  [8] 392.3190 385.9357 380.7097 376.3894 372.6606 369.3319 366.2011
## [15] 363.1763 360.0137 357.1169 354.5526 352.3620 350.5564 349.0686
## [22] 347.8362 346.8148 345.9766 345.2887 344.7214 344.2830 344.0182
## [29] 343.8935 343.7810 343.7508 343.7536 343.7974 343.8500 343.8811
## [36] 344.0443 344.3803 344.5123 344.3427 343.8007 343.1129 342.3326
## [43] 341.7577 341.3319 341.0190 340.7809 340.6375 340.2992 339.9782
## [50] 339.7093 339.5198 339.3855 339.3675 339.4130 339.4511 339.4720
## [57] 339.4860 339.6046 339.7626 339.9361 340.0813 340.1993 340.3588
## [64] 340.5591 340.7645 340.9708 341.1612 341.3457 341.5311 341.7186
## [71] 341.8968 342.0393 342.2741
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.3675
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 372.6606
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
## 1  255.2820965 204088.1  34638.53  238726.7 169449.59     0
## 2  232.6035386 197335.9  35294.25  232630.1 162041.62     1
## 3  211.9396813 188439.1  34293.40  222732.5 154145.65     2
## 4  193.1115442 180850.9  33348.79  214199.7 147502.07     2
## 5  175.9560468 173894.6  32598.47  206493.1 141296.14     3
## 6  160.3245966 166389.4  31945.17  198334.6 134444.26     4
## 7  146.0818013 159624.2  31337.42  190961.7 128286.82     4
## 8  133.1042967 153914.2  30803.46  184717.7 123110.74     4
## 9  121.2796778 148946.4  30242.19  179188.6 118704.17     4
## 10 110.5055255 144939.9  29767.40  174707.3 115172.48     4
## 11 100.6885192 141669.0  29382.16  171051.1 112286.83     5
## 12  91.7436287 138876.0  29024.19  167900.2 109851.77     5
## 13  83.5933775 136406.0  28686.95  165093.0 107719.09     5
## 14  76.1671723 134103.3  28404.31  162507.6 105698.94     5
## 15  69.4006906 131897.0  28166.25  160063.3 103730.76     6
## 16  63.2353245 129609.9  27910.61  157520.5 101699.26     6
## 17  57.6176726 127532.5  27652.48  155185.0  99880.00     6
## 18  52.4990774 125707.6  27353.14  153060.7  98354.42     6
## 19  47.8352040 124159.0  27063.77  151222.7  97095.19     6
## 20  43.5856563 122889.8  26813.65  149703.4  96076.13     6
## 21  39.7136268 121848.9  26597.85  148446.7  95251.04     6
## 22  36.1855776 120990.0  26414.07  147404.1  94575.94     6
## 23  32.9709506 120280.5  26259.56  146540.1  94020.96     6
## 24  30.0419022 119699.8  26128.74  145828.5  93571.06     6
## 25  27.3730624 119224.3  26016.33  145240.6  93207.94     6
## 26  24.9413150 118832.9  25918.40  144751.3  92914.47     6
## 27  22.7255973 118530.8  25830.43  144361.2  92700.38     6
## 28  20.7067179 118348.5  25754.37  144102.9  92594.12     6
## 29  18.8671902 118262.7  25684.49  143947.2  92578.26     6
## 30  17.1910810 118185.4  25626.78  143812.2  92558.60     7
## 31  15.6638727 118164.6  25573.78  143738.4  92590.85     7
## 32  14.2723374 118166.6  25521.00  143687.6  92645.57     7
## 33  13.0044223 118196.7  25478.22  143674.9  92718.44     9
## 34  11.8491453 118232.8  25455.46  143688.3  92777.33     9
## 35  10.7964999 118254.2  25442.84  143697.0  92811.35     9
## 36   9.8373686 118366.5  25435.44  143801.9  92931.02     9
## 37   8.9634439 118597.8  25434.49  144032.3  93163.30     9
## 38   8.1671562 118688.7  25401.65  144090.4  93287.09    11
## 39   7.4416086 118571.9  25272.79  143844.7  93299.09    11
## 40   6.7805166 118198.9  25119.84  143318.8  93079.10    12
## 41   6.1781542 117726.5  24976.44  142702.9  92750.04    12
## 42   5.6293040 117191.6  24834.88  142026.5  92356.74    13
## 43   5.1292121 116798.4  24749.54  141547.9  92048.82    13
## 44   4.6735471 116507.4  24692.28  141199.7  91815.17    13
## 45   4.2583620 116294.0  24659.61  140953.6  91634.38    13
## 46   3.8800609 116131.7  24656.53  140788.2  91475.13    13
## 47   3.5353670 116033.9  24678.64  140712.6  91355.28    13
## 48   3.2212947 115803.6  24711.53  140515.1  91092.03    13
## 49   2.9351238 115585.2  24731.68  140316.9  90853.49    13
## 50   2.6743755 115402.4  24726.94  140129.4  90675.50    13
## 51   2.4367913 115273.7  24717.01  139990.7  90556.70    13
## 52   2.2203135 115182.5  24718.79  139901.3  90463.72    14
## 53   2.0230670 115170.3  24739.86  139910.2  90430.46    15
## 54   1.8433433 115201.2  24760.18  139961.3  90440.99    15
## 55   1.6795857 115227.1  24782.17  140009.2  90444.89    17
## 56   1.5303760 115241.2  24806.68  140047.9  90434.56    17
## 57   1.3944216 115250.7  24838.86  140089.6  90411.87    17
## 58   1.2705450 115331.3  24859.15  140190.5  90472.16    17
## 59   1.1576733 115438.6  24874.54  140313.2  90564.10    17
## 60   1.0548288 115556.6  24891.15  140447.7  90665.44    17
## 61   0.9611207 115655.3  24909.43  140564.7  90745.85    17
## 62   0.8757374 115735.6  24931.84  140667.4  90803.73    17
## 63   0.7979393 115844.1  24958.68  140802.8  90885.41    17
## 64   0.7270526 115980.5  25000.92  140981.4  90979.55    17
## 65   0.6624632 116120.4  25048.65  141169.1  91071.77    18
## 66   0.6036118 116261.1  25095.01  141356.1  91166.05    18
## 67   0.5499886 116391.0  25138.96  141529.9  91252.03    18
## 68   0.5011291 116516.9  25183.29  141700.2  91333.60    17
## 69   0.4566102 116643.5  25221.26  141864.8  91422.26    18
## 70   0.4160462 116771.6  25252.15  142023.8  91519.45    18
## 71   0.3790858 116893.4  25281.92  142175.3  91611.50    18
## 72   0.3454089 116990.9  25310.23  142301.1  91680.67    18
## 73   0.3147237 117151.6  25341.58  142493.1  91809.98    18
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

![](14-shrink_files/figure-latex/unnamed-chunk-15-1.pdf)<!-- --> 


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

![](14-shrink_files/figure-latex/unnamed-chunk-19-1.pdf)<!-- --> 


```r
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```

![](14-shrink_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 

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
##  [1] "reshape2"     "splines"      "colorspace"   "htmltools"   
##  [5] "stats4"       "yaml"         "mgcv"         "rlang"       
##  [9] "ModelMetrics" "e1071"        "nloptr"       "foreign"     
## [13] "glue"         "bindrcpp"     "bindr"        "plyr"        
## [17] "stringr"      "MatrixModels" "munsell"      "gtable"      
## [21] "codetools"    "psych"        "evaluate"     "knitr"       
## [25] "SparseM"      "class"        "quantreg"     "pbkrtest"    
## [29] "parallel"     "Rcpp"         "backports"    "scales"      
## [33] "lme4"         "mnormt"       "digest"       "stringi"     
## [37] "bookdown"     "dplyr"        "grid"         "rprojroot"   
## [41] "tools"        "magrittr"     "lazyeval"     "tibble"      
## [45] "tidyr"        "car"          "pkgconfig"    "MASS"        
## [49] "assertthat"   "minqa"        "rmarkdown"    "iterators"   
## [53] "R6"           "nnet"         "nlme"         "compiler"
```
