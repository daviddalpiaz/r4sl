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
## (Intercept) 226.844379273
## AtBat         0.086613903
## Hits          0.352962516
## HmRun         1.144213853
## Runs          0.569353374
## RBI           0.570074068
## Walks         0.735072620
## Years         2.397356093
## CAtBat        0.007295083
## CHits         0.027995153
## CHmRun        0.208112350
## CRuns         0.056146220
## CRBI          0.058060281
## CWalks        0.056586702
## LeagueN       2.850306112
## DivisionW   -20.329125702
## PutOuts       0.049296951
## Assists       0.007063169
## Errors       -0.128066381
## NewLeagueN    2.654025563
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
## (Intercept) 226.844379273
## AtBat         0.086613903
## Hits          0.352962516
## HmRun         1.144213853
## Runs          0.569353374
## RBI           0.570074068
## Walks         0.735072620
## Years         2.397356093
## CAtBat        0.007295083
## CHits         0.027995153
## CHmRun        0.208112350
## CRuns         0.056146220
## CRBI          0.058060281
## CWalks        0.056586702
## LeagueN       2.850306112
## DivisionW   -20.329125702
## PutOuts       0.049296951
## Assists       0.007063169
## Errors       -0.128066381
## NewLeagueN    2.654025563
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 436.8923
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 134397.5
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.1699 448.9269 448.5197 448.2498 447.9548 447.6325 447.2804
##  [8] 446.8960 446.4765 446.0190 445.5203 444.9772 444.3859 443.7429
## [15] 443.0440 442.2851 441.4621 440.5705 439.6055 438.5627 437.4373
## [22] 436.2245 434.9199 433.5189 432.0174 430.4114 428.6978 426.8736
## [29] 424.9368 422.8864 420.7222 418.4452 416.0578 413.5638 410.9687
## [36] 408.2794 405.5051 402.6554 399.7428 396.7807 393.7842 390.7694
## [43] 387.7531 384.7529 381.7864 378.8710 376.0234 373.2593 370.5929
## [50] 368.0368 365.6015 363.2957 361.1253 359.0952 357.2073 355.4614
## [57] 353.8559 352.3873 351.0511 349.8414 348.7518 347.7752 346.9042
## [64] 346.1312 345.4493 344.8556 344.3361 343.8862 343.4996 343.1720
## [71] 342.8954 342.6678 342.4779 342.3256 342.2022 342.1080 342.0341
## [78] 341.9782 341.9362 341.9032 341.8788 341.8563 341.8356 341.8153
## [85] 341.7847 341.7538 341.7130 341.6600 341.5997 341.5274 341.4447
## [92] 341.3476 341.2416 341.1224 340.9943 340.8541 340.7087 340.5497
## [99] 340.3932
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.3932
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.5929
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
##  [1] 450.7586 442.9704 432.9210 424.4450 417.1364 409.2731 401.3610
##  [8] 394.2792 387.8634 382.1884 377.5396 373.6263 370.2357 367.3076
## [15] 364.5395 361.7850 359.1002 356.4940 354.3076 352.4843 350.9643
## [22] 349.6992 348.6478 347.8175 347.2071 346.7443 346.3702 346.0781
## [29] 345.8895 345.7498 345.6366 345.5643 345.5237 345.4907 345.4682
## [36] 345.5443 345.7783 346.1663 346.5029 346.5347 346.2164 345.6549
## [43] 345.1858 344.7971 344.4382 344.1275 343.7418 343.2164 342.8960
## [50] 342.7734 342.9404 343.3346 343.8315 344.3348 344.7957 345.0692
## [57] 345.2286 345.3703 345.5007 345.6138 345.7494 345.8910 346.0609
## [64] 346.2504 346.4514 346.6047 346.7591 346.8447 346.9422 347.0700
## [71] 347.1298 347.1829
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.7734
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.2357
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
## 1  255.2820965 203183.3  31011.02  234194.3 172172.29     0
## 2  232.6035386 196222.7  31295.99  227518.7 164926.75     1
## 3  211.9396813 187420.6  30002.56  217423.2 157418.03     2
## 4  193.1115442 180153.6  28789.55  208943.1 151364.03     2
## 5  175.9560468 174002.8  27755.99  201758.8 146246.81     3
## 6  160.3245966 167504.5  26836.65  194341.1 140667.82     4
## 7  146.0818013 161090.6  26031.72  187122.4 135058.91     4
## 8  133.1042967 155456.1  25360.55  180816.6 130095.51     4
## 9  121.2796778 150438.0  24743.01  175181.0 125694.99     4
## 10 110.5055255 146068.0  24101.98  170170.0 121966.00     4
## 11 100.6885192 142536.2  23609.75  166145.9 118926.41     5
## 12  91.7436287 139596.6  23234.61  162831.2 116361.97     5
## 13  83.5933775 137074.5  22913.10  159987.6 114161.35     5
## 14  76.1671723 134914.8  22643.41  157558.2 112271.43     5
## 15  69.4006906 132889.1  22475.06  155364.1 110414.00     6
## 16  63.2353245 130888.4  22388.77  153277.1 108499.58     6
## 17  57.6176726 128952.9  22223.91  151176.9 106729.04     6
## 18  52.4990774 127088.0  21986.84  149074.8 105101.13     6
## 19  47.8352040 125533.9  21797.80  147331.7 103736.08     6
## 20  43.5856563 124245.2  21650.33  145895.5 102594.87     6
## 21  39.7136268 123175.9  21535.46  144711.4 101640.47     6
## 22  36.1855776 122289.6  21447.15  143736.7 100842.41     6
## 23  32.9709506 121555.3  21380.36  142935.6 100174.93     6
## 24  30.0419022 120977.0  21345.30  142322.3  99631.71     6
## 25  27.3730624 120552.8  21350.30  141903.1  99202.46     6
## 26  24.9413150 120231.6  21374.53  141606.2  98857.11     6
## 27  22.7255973 119972.3  21403.50  141375.8  98568.83     6
## 28  20.7067179 119770.0  21435.82  141205.8  98334.21     6
## 29  18.8671902 119639.5  21467.77  141107.3  98171.77     6
## 30  17.1910810 119542.9  21498.37  141041.3  98044.58     7
## 31  15.6638727 119464.7  21527.32  140992.0  97937.34     7
## 32  14.2723374 119414.7  21560.78  140975.5  97853.91     7
## 33  13.0044223 119386.6  21599.60  140986.2  97787.01     9
## 34  11.8491453 119363.8  21636.22  141000.0  97727.59     9
## 35  10.7964999 119348.2  21670.64  141018.9  97677.61     9
## 36   9.8373686 119400.9  21699.39  141100.3  97701.49     9
## 37   8.9634439 119562.6  21702.77  141265.4  97859.88     9
## 38   8.1671562 119831.1  21673.94  141505.0  98157.17    11
## 39   7.4416086 120064.3  21669.92  141734.2  98394.34    11
## 40   6.7805166 120086.3  21588.29  141674.6  98498.04    12
## 41   6.1781542 119865.8  21402.73  141268.5  98463.08    12
## 42   5.6293040 119477.3  21270.94  140748.2  98206.37    13
## 43   5.1292121 119153.2  21162.13  140315.3  97991.08    13
## 44   4.6735471 118885.1  21074.48  139959.5  97810.60    13
## 45   4.2583620 118637.7  20998.52  139636.2  97639.18    13
## 46   3.8800609 118423.7  20923.63  139347.4  97500.10    13
## 47   3.5353670 118158.5  20845.18  139003.6  97313.28    13
## 48   3.2212947 117797.5  20759.21  138556.7  97038.27    13
## 49   2.9351238 117577.7  20687.36  138265.0  96890.30    13
## 50   2.6743755 117493.6  20614.61  138108.2  96879.01    13
## 51   2.4367913 117608.1  20542.30  138150.4  97065.82    13
## 52   2.2203135 117878.6  20474.58  138353.2  97404.05    14
## 53   2.0230670 118220.1  20414.93  138635.0  97805.15    15
## 54   1.8433433 118566.5  20378.33  138944.8  98188.12    15
## 55   1.6795857 118884.1  20361.14  139245.2  98522.92    17
## 56   1.5303760 119072.8  20349.71  139422.5  98723.05    17
## 57   1.3944216 119182.8  20354.15  139536.9  98828.65    17
## 58   1.2705450 119280.6  20369.37  139650.0  98911.27    17
## 59   1.1576733 119370.8  20374.13  139744.9  98996.63    17
## 60   1.0548288 119448.9  20379.80  139828.7  99069.09    17
## 61   0.9611207 119542.6  20390.66  139933.3  99151.97    17
## 62   0.8757374 119640.6  20404.38  140045.0  99236.22    17
## 63   0.7979393 119758.2  20422.25  140180.4  99335.92    17
## 64   0.7270526 119889.3  20440.32  140329.7  99449.02    17
## 65   0.6624632 120028.5  20461.23  140489.8  99567.31    18
## 66   0.6036118 120134.8  20476.43  140611.3  99658.41    18
## 67   0.5499886 120241.8  20477.97  140719.8  99763.87    18
## 68   0.5011291 120301.3  20482.85  140784.1  99818.43    17
## 69   0.4566102 120368.9  20479.57  140848.4  99889.29    18
## 70   0.4160462 120457.6  20478.33  140935.9  99979.24    18
## 71   0.3790858 120499.1  20482.16  140981.3 100016.95    18
## 72   0.3454089 120536.0  20480.17  141016.1 100055.78    18
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
