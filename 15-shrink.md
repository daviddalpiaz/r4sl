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
##  [1] 452.0562 450.8421 449.7409 449.4753 449.1850 448.8678 448.5213
##  [8] 448.1429 447.7300 447.2797 446.7889 446.2542 445.6722 445.0391
## [15] 444.3511 443.6039 442.7935 441.9154 440.9651 439.9380 438.8294
## [22] 437.6346 436.3491 434.9685 433.4885 431.9054 430.2158 428.4167
## [29] 426.5063 424.4831 422.3470 420.0989 417.7409 415.2767 412.7115
## [36] 410.0519 407.3066 404.4854 401.5998 398.6632 395.6901 392.6961
## [43] 389.6979 386.7125 383.7571 380.8488 378.0040 375.2381 372.5656
## [50] 369.9979 367.5461 365.2184 363.0217 360.9602 359.0358 357.2487
## [57] 355.5973 354.0783 352.6874 351.4188 350.2664 349.2232 348.2821
## [64] 347.4340 346.6780 346.0034 345.4028 344.8691 344.3957 343.9795
## [71] 343.6170 343.3009 343.0227 342.7818 342.5748 342.3943 342.2401
## [78] 342.1034 341.9880 341.8852 341.7957 341.7134 341.6388 341.5679
## [85] 341.4992 341.4324 341.3628 341.2912 341.2181 341.1411 341.0601
## [92] 340.9758 340.8852 340.7915 340.6941 340.5948 340.4940 340.3922
## [99] 340.2921
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.2921
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.5461
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
## (Intercept)  134.48030406
## AtBat         -1.67572220
## Hits           5.94122316
## HmRun          0.04746835
## Runs           .         
## RBI            .         
## Walks          4.95676182
## Years        -10.26657309
## CAtBat         .         
## CHits          .         
## CHmRun         0.56236426
## CRuns          0.70135135
## CRBI           0.38727139
## CWalks        -0.58111548
## LeagueN       32.92255640
## DivisionW   -119.37941356
## PutOuts        0.27580087
## Assists        0.19782326
## Errors        -2.26242857
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 180.1579
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
##  [1] 449.1210 439.0433 428.5051 419.7046 411.7497 403.4277 394.9349
##  [8] 387.3061 380.7830 375.2311 370.6526 366.8440 363.7734 360.9544
## [15] 358.1254 355.1814 352.4209 350.0121 347.9642 346.2705 344.8779
## [22] 343.7449 342.8162 342.0645 341.4953 341.0739 340.7515 340.5088
## [29] 340.3500 340.2689 340.2386 340.2199 340.1831 340.1632 340.1563
## [36] 340.2321 340.3388 340.2376 339.7276 338.9435 338.1314 337.3837
## [43] 336.7045 335.9926 335.2707 334.7146 334.3283 333.9446 333.6466
## [50] 333.4256 333.2875 333.2807 333.3894 333.4250 333.4305 333.4542
## [57] 333.5812 333.7527 334.0350 334.2869 334.5266 334.7653 335.0036
## [64] 335.2169 335.3802 335.6123 335.7644 335.9021 336.0539 336.1575
## [71] 336.2790 336.3850 336.5099 336.5966 336.6696
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 333.2807
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 360.9544
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
## 1  255.2820965 201709.7  11435.76  213145.4 190273.92     0
## 2  232.6035386 192759.0  11728.19  204487.2 181030.80     1
## 3  211.9396813 183616.6  12088.39  195705.0 171528.25     2
## 4  193.1115442 176151.9  12521.33  188673.3 163630.61     2
## 5  175.9560468 169537.8  13177.95  182715.8 156359.90     3
## 6  160.3245966 162753.9  13984.14  176738.0 148769.75     4
## 7  146.0818013 155973.6  14704.71  170678.3 141268.84     4
## 8  133.1042967 150006.0  15332.30  165338.3 134673.72     4
## 9  121.2796778 144995.7  15896.30  160892.0 129099.42     4
## 10 110.5055255 140798.4  16427.14  157225.5 124371.24     4
## 11 100.6885192 137383.4  16928.53  154311.9 120454.85     5
## 12  91.7436287 134574.5  17390.64  151965.1 117183.85     5
## 13  83.5933775 132331.1  17815.30  150146.4 114515.82     5
## 14  76.1671723 130288.1  18185.75  148473.8 112102.31     5
## 15  69.4006906 128253.8  18540.90  146794.7 109712.92     6
## 16  63.2353245 126153.8  18904.64  145058.5 107249.18     6
## 17  57.6176726 124200.5  19211.08  143411.6 104989.42     6
## 18  52.4990774 122508.5  19413.43  141921.9 103095.04     6
## 19  47.8352040 121079.1  19569.46  140648.5 101509.61     6
## 20  43.5856563 119903.2  19717.29  139620.5 100185.96     6
## 21  39.7136268 118940.8  19858.52  138799.3  99082.25     6
## 22  36.1855776 118160.6  19994.92  138155.5  98165.67     6
## 23  32.9709506 117523.0  20124.09  137647.1  97398.88     6
## 24  30.0419022 117008.1  20245.05  137253.2  96763.06     6
## 25  27.3730624 116619.0  20354.05  136973.1  96264.99     6
## 26  24.9413150 116331.4  20452.49  136783.9  95878.89     6
## 27  22.7255973 116111.6  20543.92  136655.5  95567.63     6
## 28  20.7067179 115946.2  20625.36  136571.6  95320.85     6
## 29  18.8671902 115838.1  20703.03  136541.1  95135.08     6
## 30  17.1910810 115782.9  20772.11  136555.0  95010.82     7
## 31  15.6638727 115762.3  20836.00  136598.3  94926.32     7
## 32  14.2723374 115749.6  20890.45  136640.1  94859.15     7
## 33  13.0044223 115724.6  20937.87  136662.4  94786.68     9
## 34  11.8491453 115711.0  20981.64  136692.6  94729.36     9
## 35  10.7964999 115706.3  21023.78  136730.1  94682.50     9
## 36   9.8373686 115757.9  21057.00  136814.9  94700.88     9
## 37   8.9634439 115830.5  21087.98  136918.5  94742.54     9
## 38   8.1671562 115761.6  21073.48  136835.1  94688.15    11
## 39   7.4416086 115414.9  21014.59  136429.4  94400.27    11
## 40   6.7805166 114882.7  20945.41  135828.1  93937.26    12
## 41   6.1781542 114332.9  20838.66  135171.5  93494.21    12
## 42   5.6293040 113827.7  20731.68  134559.4  93096.06    13
## 43   5.1292121 113370.0  20630.51  134000.5  92739.44    13
## 44   4.6735471 112891.0  20536.90  133427.9  92354.09    13
## 45   4.2583620 112406.5  20452.33  132858.8  91954.14    13
## 46   3.8800609 112033.8  20384.30  132418.1  91649.55    13
## 47   3.5353670 111775.4  20314.54  132090.0  91460.91    13
## 48   3.2212947 111519.0  20229.00  131748.0  91289.98    13
## 49   2.9351238 111320.0  20157.92  131478.0  91162.12    13
## 50   2.6743755 111172.6  20094.88  131267.5  91077.76    13
## 51   2.4367913 111080.6  20048.32  131128.9  91032.24    13
## 52   2.2203135 111076.0  20030.42  131106.5  91045.62    14
## 53   2.0230670 111148.5  20028.44  131177.0  91120.07    15
## 54   1.8433433 111172.2  20037.41  131209.6  91134.83    15
## 55   1.6795857 111175.9  20055.34  131231.2  91120.53    17
## 56   1.5303760 111191.7  20089.00  131280.7  91102.71    17
## 57   1.3944216 111276.4  20131.13  131407.5  91145.26    17
## 58   1.2705450 111390.9  20176.31  131567.2  91214.58    17
## 59   1.1576733 111579.4  20224.33  131803.7  91355.04    17
## 60   1.0548288 111747.7  20275.46  132023.2  91472.24    17
## 61   0.9611207 111908.0  20325.24  132233.3  91582.80    17
## 62   0.8757374 112067.8  20373.27  132441.1  91694.53    17
## 63   0.7979393 112227.4  20418.94  132646.4  91808.49    17
## 64   0.7270526 112370.4  20455.69  132826.1  91914.67    17
## 65   0.6624632 112479.9  20484.51  132964.4  91995.34    18
## 66   0.6036118 112635.6  20508.46  133144.1  92127.14    18
## 67   0.5499886 112737.7  20532.10  133269.8  92205.61    18
## 68   0.5011291 112830.2  20554.85  133385.0  92275.34    17
## 69   0.4566102 112932.2  20573.21  133505.5  92359.03    18
## 70   0.4160462 113001.9  20594.78  133596.6  92407.08    18
## 71   0.3790858 113083.6  20611.40  133695.0  92472.16    18
## 72   0.3454089 113154.8  20631.65  133786.5  92523.19    18
## 73   0.3147237 113238.9  20654.99  133893.9  92583.93    18
## 74   0.2867645 113297.3  20679.71  133977.0  92617.59    18
## 75   0.2612891 113346.4  20702.42  134048.8  92643.99    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.220313   76.16717
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
