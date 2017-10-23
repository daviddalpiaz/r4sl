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
##  [1] 451.6635 449.9307 449.2257 448.9564 448.6620 448.3403 447.9890
##  [8] 447.6055 447.1870 446.7306 446.2333 445.6916 445.1020 444.4609
## [15] 443.7643 443.0079 442.1878 441.2995 440.3385 439.3001 438.1797
## [22] 436.9729 435.6750 434.2817 432.7891 431.1934 429.4914 427.6807
## [29] 425.7593 423.7263 421.5820 419.3276 416.9657 414.5006 411.9378
## [36] 409.2849 406.5508 403.7459 400.8826 397.9747 395.0372 392.0864
## [43] 389.1391 386.2127 383.3244 380.4914 377.7300 375.0551 372.4808
## [50] 370.0182 367.6763 365.4640 363.3862 361.4460 359.6442 357.9800
## [57] 356.4504 355.0513 353.7769 352.6210 351.5762 350.6351 349.7895
## [64] 349.0284 348.3525 347.7509 347.2175 346.7413 346.3117 345.9382
## [71] 345.6080 345.3103 345.0481 344.8176 344.6035 344.4141 344.2370
## [78] 344.0762 343.9238 343.7768 343.6296 343.4920 343.3427 343.1991
## [85] 343.0410 342.8875 342.7055 342.5368 342.3363 342.1508 341.9212
## [92] 341.7177 341.4636 341.2421 340.9683 340.7317 340.4433 340.1979
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.1979
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.6763
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
## (Intercept)  150.91444426
## AtBat         -1.90103581
## Hits           6.65130985
## HmRun          1.04183116
## Runs          -0.76385256
## RBI            .         
## Walks          5.50826655
## Years         -8.20078865
## CAtBat        -0.05363154
## CHits          .         
## CHmRun         0.29763955
## CRuns          1.05083933
## CRBI           0.53509241
## CWalks        -0.70810651
## LeagueN       44.17286120
## DivisionW   -117.10970367
## PutOuts        0.28149966
## Assists        0.27624774
## Errors        -2.77650223
## NewLeagueN    -8.10900042
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 199.4382
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
##  [1] 450.0162 441.3141 430.9058 422.0097 413.6434 404.7826 396.3554
##  [8] 388.6859 382.2563 376.9474 372.6192 369.0348 366.0425 363.4197
## [15] 360.7041 357.5846 354.7424 352.2644 349.9998 348.0844 346.4964
## [22] 345.1691 344.0567 343.1315 342.3735 341.7454 341.2609 340.9216
## [29] 340.6990 340.6399 340.6234 340.6599 340.6989 340.8463 341.1479
## [36] 341.5860 342.1960 342.8814 343.3281 343.0680 342.6458 342.1349
## [43] 341.1711 340.1618 339.3636 338.6929 338.1014 337.6695 337.3550
## [50] 337.1496 336.9991 336.9148 336.9617 337.0433 337.1054 337.0574
## [57] 336.9246 336.7996 336.7267 336.7433 336.8129 336.9424 337.1091
## [64] 337.2488 337.3781 337.4898 337.6353 337.7655 337.8576 337.9701
## [71] 338.0865 338.1803
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.7267
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 360.7041
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
## 1  255.2820965 202514.6  30566.18  233080.8 171948.41     0
## 2  232.6035386 194758.1  30188.28  224946.4 164569.87     1
## 3  211.9396813 185679.8  29171.21  214851.0 156508.56     2
## 4  193.1115442 178092.2  28379.02  206471.2 149713.21     2
## 5  175.9560468 171100.8  27669.88  198770.7 143430.96     3
## 6  160.3245966 163849.0  27066.77  190915.8 136782.22     4
## 7  146.0818013 157097.6  26608.01  183705.6 130489.64     4
## 8  133.1042967 151076.7  26212.21  177288.9 124864.50     4
## 9  121.2796778 146119.9  25863.45  171983.3 120256.41     4
## 10 110.5055255 142089.3  25474.04  167563.4 116615.29     4
## 11 100.6885192 138845.1  25150.33  163995.4 113694.77     5
## 12  91.7436287 136186.7  24888.43  161075.1 111298.26     5
## 13  83.5933775 133987.1  24676.75  158663.8 109310.34     5
## 14  76.1671723 132073.9  24506.66  156580.5 107567.21     5
## 15  69.4006906 130107.4  24309.71  154417.1 105797.71     6
## 16  63.2353245 127866.8  23885.74  151752.5 103981.04     6
## 17  57.6176726 125842.2  23504.57  149346.7 102337.60     6
## 18  52.4990774 124090.2  23086.33  147176.6 101003.91     6
## 19  47.8352040 122499.8  22580.03  145079.9  99919.81     6
## 20  43.5856563 121162.8  22138.37  143301.1  99024.38     6
## 21  39.7136268 120059.8  21765.57  141825.3  98294.19     6
## 22  36.1855776 119141.7  21451.56  140593.2  97690.12     6
## 23  32.9709506 118375.0  21186.89  139561.9  97188.16     6
## 24  30.0419022 117739.2  20967.65  138706.9  96771.59     6
## 25  27.3730624 117219.6  20787.01  138006.6  96432.57     6
## 26  24.9413150 116789.9  20632.96  137422.9  96156.96     6
## 27  22.7255973 116459.0  20506.15  136965.1  95952.85     6
## 28  20.7067179 116227.5  20398.85  136626.4  95828.68     6
## 29  18.8671902 116075.8  20324.11  136399.9  95751.72     6
## 30  17.1910810 116035.6  20327.11  136362.7  95708.45     7
## 31  15.6638727 116024.3  20343.85  136368.2  95680.46     7
## 32  14.2723374 116049.2  20359.28  136408.5  95689.89     7
## 33  13.0044223 116075.7  20367.36  136443.1  95708.36     9
## 34  11.8491453 116176.2  20376.99  136553.2  95799.18     9
## 35  10.7964999 116381.9  20379.29  136761.2  96002.62     9
## 36   9.8373686 116681.0  20360.04  137041.0  96320.93     9
## 37   8.9634439 117098.1  20320.16  137418.3  96777.94     9
## 38   8.1671562 117567.7  20274.91  137842.6  97292.75    11
## 39   7.4416086 117874.2  20274.96  138149.1  97599.22    11
## 40   6.7805166 117695.7  20203.14  137898.8  97492.52    12
## 41   6.1781542 117406.1  20144.60  137550.8  97261.54    12
## 42   5.6293040 117056.3  20098.31  137154.6  96957.97    13
## 43   5.1292121 116397.7  19776.86  136174.6  96620.89    13
## 44   4.6735471 115710.1  19437.93  135148.0  96272.12    13
## 45   4.2583620 115167.6  19187.10  134354.7  95980.53    13
## 46   3.8800609 114712.9  18945.58  133658.5  95767.33    13
## 47   3.5353670 114312.5  18701.16  133013.7  95611.38    13
## 48   3.2212947 114020.7  18481.15  132501.8  95539.53    13
## 49   2.9351238 113808.4  18280.21  132088.6  95528.17    13
## 50   2.6743755 113669.9  18122.87  131792.7  95547.00    13
## 51   2.4367913 113568.4  17969.63  131538.1  95598.79    13
## 52   2.2203135 113511.6  17835.21  131346.8  95676.40    14
## 53   2.0230670 113543.2  17712.86  131256.1  95830.34    15
## 54   1.8433433 113598.2  17603.98  131202.1  95994.18    15
## 55   1.6795857 113640.0  17506.59  131146.6  96133.44    17
## 56   1.5303760 113607.7  17348.76  130956.4  96258.91    17
## 57   1.3944216 113518.2  17239.75  130757.9  96278.44    17
## 58   1.2705450 113434.0  17165.63  130599.6  96268.34    17
## 59   1.1576733 113384.8  17125.85  130510.7  96259.00    17
## 60   1.0548288 113396.1  17107.47  130503.5  96288.60    17
## 61   0.9611207 113442.9  17103.73  130546.6  96339.17    17
## 62   0.8757374 113530.2  17105.96  130636.1  96424.20    17
## 63   0.7979393 113642.5  17115.73  130758.3  96526.81    17
## 64   0.7270526 113736.8  17106.86  130843.6  96629.90    17
## 65   0.6624632 113824.0  17090.18  130914.2  96733.79    18
## 66   0.6036118 113899.4  17080.93  130980.3  96818.44    18
## 67   0.5499886 113997.6  17073.42  131071.0  96924.19    18
## 68   0.5011291 114085.5  17069.56  131155.1  97015.96    17
## 69   0.4566102 114147.8  17070.62  131218.4  97077.14    18
## 70   0.4160462 114223.8  17070.73  131294.5  97153.06    18
## 71   0.3790858 114302.5  17071.14  131373.6  97231.34    18
## 72   0.3454089 114365.9  17073.68  131439.6  97292.21    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   1.157673   69.40069
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
