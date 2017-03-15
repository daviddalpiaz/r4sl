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
##  [1] 451.7389 450.0284 449.3288 449.0601 448.7664 448.4455 448.0949
##  [8] 447.7123 447.2947 446.8394 446.3431 445.8026 445.2142 444.5744
## [15] 443.8792 443.1243 442.3058 441.4191 440.4598 439.4231 438.3046
## [22] 437.0995 435.8034 434.4119 432.9209 431.3267 429.6261 427.8164
## [29] 425.8957 423.8630 421.7183 419.4628 417.0989 414.6307 412.0636
## [36] 409.4047 406.6629 403.8483 400.9729 398.0502 395.0951 392.1232
## [43] 389.1513 386.1964 383.2756 380.4056 377.6027 374.8819 372.2569
## [50] 369.7391 367.3377 365.0617 362.9164 360.9053 359.0297 357.2892
## [57] 355.6814 354.2028 352.8483 351.6120 350.4874 349.4674 348.5447
## [64] 347.7094 346.9600 346.2881 345.6869 345.1419 344.6572 344.2273
## [71] 343.8425 343.4942 343.1844 342.9041 342.6550 342.4317 342.2218
## [78] 342.0330 341.8557 341.6847 341.5304 341.3723 341.2211 341.0729
## [85] 340.9206 340.7654 340.6076 340.4468 340.2815 340.1087 339.9303
## [92] 339.7468 339.5593 339.3653 339.1677 338.9678 338.7648 338.5619
## [99] 338.3612
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.3612
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 360.9053
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
##  [1] 451.2181 441.8008 431.7687 423.2147 415.2184 406.9010 398.5968
##  [8] 391.0525 384.5780 379.1099 374.4393 370.0714 366.0301 362.4465
## [15] 359.1806 356.0587 353.1419 350.7045 348.6563 346.9099 345.4508
## [22] 344.2601 343.2740 342.4787 341.9234 341.5021 341.1809 340.9940
## [29] 340.8875 340.8752 340.8917 340.9281 340.9422 340.9817 341.1219
## [36] 341.4382 342.0483 342.6944 343.0827 343.2587 343.2104 342.8888
## [43] 342.2953 341.8046 341.3996 340.8236 340.3932 340.0868 339.8584
## [50] 339.7766 339.8798 340.2723 340.7832 341.3359 341.8796 342.4266
## [57] 342.9027 343.2815 343.5282 343.7488 343.9403 344.1562 344.3361
## [64] 344.4801 344.6083 344.7800 344.8855 345.0147 345.1527 345.3100
## [71] 345.4062 345.5663 345.6282
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.7766
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.0714
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
## 1  255.2820965 203597.8  24378.99  227976.8 179218.80     0
## 2  232.6035386 195188.0  23958.80  219146.8 171229.19     1
## 3  211.9396813 186424.2  23103.76  209528.0 163320.43     2
## 4  193.1115442 179110.7  22373.63  201484.3 156737.07     2
## 5  175.9560468 172406.3  21936.06  194342.4 150470.24     3
## 6  160.3245966 165568.4  21686.44  187254.9 143881.98     4
## 7  146.0818013 158879.4  21487.15  180366.6 137392.27     4
## 8  133.1042967 152922.0  21378.42  174300.4 131543.60     4
## 9  121.2796778 147900.3  21406.97  169307.2 126493.28     4
## 10 110.5055255 143724.3  21517.17  165241.5 122207.12     4
## 11 100.6885192 140204.8  21689.60  161894.4 118515.17     5
## 12  91.7436287 136952.8  21820.64  158773.5 115132.19     5
## 13  83.5933775 133978.0  21984.79  155962.8 111993.21     5
## 14  76.1671723 131367.5  22197.77  153565.3 109169.72     5
## 15  69.4006906 129010.7  22429.89  151440.6 106580.79     6
## 16  63.2353245 126777.8  22594.70  149372.5 104183.07     6
## 17  57.6176726 124709.2  22731.35  147440.5 101977.84     6
## 18  52.4990774 122993.6  22864.53  145858.2 100129.11     6
## 19  47.8352040 121561.2  22991.64  144552.8  98569.56     6
## 20  43.5856563 120346.5  23120.65  143467.1  97225.85     6
## 21  39.7136268 119336.3  23254.97  142591.3  96081.31     6
## 22  36.1855776 118515.0  23387.40  141902.4  95127.62     6
## 23  32.9709506 117837.0  23514.60  141351.6  94322.43     6
## 24  30.0419022 117291.6  23633.18  140924.8  93658.45     6
## 25  27.3730624 116911.6  23739.59  140651.2  93172.00     6
## 26  24.9413150 116623.7  23838.17  140461.9  92785.53     6
## 27  22.7255973 116404.4  23935.28  140339.7  92469.10     6
## 28  20.7067179 116276.9  24046.99  140323.9  92229.89     6
## 29  18.8671902 116204.3  24154.06  140358.3  92050.22     6
## 30  17.1910810 116195.9  24254.69  140450.6  91941.22     7
## 31  15.6638727 116207.2  24342.88  140550.0  91864.28     7
## 32  14.2723374 116232.0  24422.39  140654.4  91809.59     7
## 33  13.0044223 116241.6  24491.99  140733.6  91749.62     9
## 34  11.8491453 116268.6  24549.10  140817.7  91719.45     9
## 35  10.7964999 116364.2  24595.60  140959.8  91768.57     9
## 36   9.8373686 116580.1  24641.87  141221.9  91938.20     9
## 37   8.9634439 116997.0  24683.57  141680.6  92313.47     9
## 38   8.1671562 117439.4  24722.69  142162.1  92716.73    11
## 39   7.4416086 117705.7  24757.77  142463.5  92947.95    11
## 40   6.7805166 117826.5  24789.44  142615.9  93037.06    12
## 41   6.1781542 117793.4  24827.78  142621.2  92965.63    12
## 42   5.6293040 117572.7  24883.82  142456.5  92688.90    13
## 43   5.1292121 117166.1  24788.77  141954.9  92377.33    13
## 44   4.6735471 116830.4  24667.87  141498.3  92162.52    13
## 45   4.2583620 116553.7  24569.69  141123.4  91983.97    13
## 46   3.8800609 116160.7  24317.40  140478.1  91843.32    13
## 47   3.5353670 115867.6  24093.92  139961.5  91773.63    13
## 48   3.2212947 115659.0  23878.62  139537.6  91780.39    13
## 49   2.9351238 115503.7  23645.44  139149.2  91858.27    13
## 50   2.6743755 115448.1  23427.63  138875.7  92020.49    13
## 51   2.4367913 115518.3  23236.32  138754.6  92281.93    13
## 52   2.2203135 115785.3  23034.56  138819.8  92750.69    14
## 53   2.0230670 116133.2  22833.19  138966.4  93299.98    15
## 54   1.8433433 116510.2  22646.65  139156.8  93863.53    15
## 55   1.6795857 116881.7  22481.17  139362.9  94400.52    17
## 56   1.5303760 117256.0  22334.77  139590.7  94921.21    17
## 57   1.3944216 117582.2  22204.53  139786.8  95377.71    17
## 58   1.2705450 117842.2  22093.57  139935.7  95748.60    17
## 59   1.1576733 118011.6  21967.03  139978.6  96044.58    17
## 60   1.0548288 118163.2  21867.46  140030.7  96295.78    17
## 61   0.9611207 118294.9  21780.79  140075.7  96514.11    17
## 62   0.8757374 118443.5  21698.04  140141.5  96745.45    17
## 63   0.7979393 118567.3  21625.88  140193.2  96941.46    17
## 64   0.7270526 118666.5  21550.79  140217.3  97115.74    17
## 65   0.6624632 118754.9  21480.97  140235.9  97273.93    18
## 66   0.6036118 118873.3  21417.69  140291.0  97455.58    18
## 67   0.5499886 118946.0  21355.58  140301.6  97590.45    18
## 68   0.5011291 119035.1  21307.08  140342.2  97728.06    17
## 69   0.4566102 119130.4  21260.96  140391.3  97869.40    18
## 70   0.4160462 119239.0  21237.54  140476.6  98001.47    18
## 71   0.3790858 119305.5  21187.38  140492.8  98118.08    18
## 72   0.3454089 119416.0  21180.04  140596.1  98236.01    18
## 73   0.3147237 119458.9  21127.29  140586.2  98331.58    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   91.74363
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
