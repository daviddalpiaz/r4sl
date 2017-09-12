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
##  [1] 452.7289 451.3518 450.4254 450.1582 449.8661 449.5469 449.1983
##  [8] 448.8178 448.4026 447.9498 447.4563 446.9188 446.3339 445.6978
## [15] 445.0066 444.2562 443.4424 442.5611 441.6075 440.5772 439.4656
## [22] 438.2681 436.9803 435.5978 434.1167 432.5333 430.8444 429.0475
## [29] 427.1406 425.1230 422.9946 420.7568 418.4122 415.9647 413.4199
## [36] 410.7852 408.0693 405.2825 402.4370 399.5463 396.6252 393.6897
## [43] 390.7564 387.8425 384.9651 382.1412 379.3868 376.7171 374.1458
## [50] 371.6846 369.3428 367.1289 365.0489 363.1061 361.3020 359.6362
## [57] 358.1066 356.7095 355.4402 354.2928 353.2608 352.3374 351.5153
## [64] 350.7847 350.1431 349.5864 349.1028 348.6845 348.3229 348.0206
## [71] 347.7726 347.5698 347.4020 347.2751 347.1789 347.1106 347.0625
## [78] 347.0372 347.0284 347.0327 347.0466 347.0664 347.0924 347.1179
## [85] 347.1444 347.1689 347.1886 347.2021 347.2099 347.2104 347.2025
## [92] 347.1847 347.1586 347.1238 347.0798 347.0274 346.9678 346.9006
## [99] 346.8283
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 346.8283
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 374.1458
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
##                       1
## (Intercept) 115.3773590
## AtBat         .        
## Hits          1.4753071
## HmRun         .        
## Runs          .        
## RBI           .        
## Walks         1.6566947
## Years         .        
## CAtBat        .        
## CHits         .        
## CHmRun        .        
## CRuns         0.1660465
## CRBI          0.3453397
## CWalks        .        
## LeagueN       .        
## DivisionW   -19.2435216
## PutOuts       0.1000068
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
##                       1
## (Intercept) 115.3773590
## AtBat         .        
## Hits          1.4753071
## HmRun         .        
## Runs          .        
## RBI           .        
## Walks         1.6566947
## Years         .        
## CAtBat        .        
## CHits         .        
## CHmRun        .        
## CRuns         0.1660465
## CRBI          0.3453397
## CWalks        .        
## LeagueN       .        
## DivisionW   -19.2435216
## PutOuts       0.1000068
## Assists       .        
## Errors        .        
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 22.98692
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 116096.9
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 450.2199 441.9013 431.6078 422.9399 415.0065 406.5329 398.1911
##  [8] 390.8489 384.5277 379.2569 374.8456 371.0727 367.7153 364.6698
## [15] 361.6445 358.5480 355.5883 352.9851 350.8179 349.0121 347.5221
## [22] 346.2924 345.2748 344.4351 343.7501 343.2518 342.8837 342.6677
## [29] 342.5784 342.5697 342.5754 342.6008 342.6452 342.6597 342.8354
## [36] 343.1082 343.5989 344.0655 344.0947 343.6853 343.1096 342.5095
## [43] 341.6576 340.6860 339.8853 339.2833 338.8088 338.4477 338.2361
## [50] 338.1169 338.0397 338.0784 338.1585 338.1951 338.2649 338.3577
## [57] 338.5132 338.6369 338.7089 338.8843 339.1623 339.4699 339.7800
## [64] 340.0760 340.3544 340.5668 340.7869 340.9309 341.1172 341.2687
## [71] 341.4517
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.0397
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 358.548
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
## 1  255.2820965 202697.9  27129.75  229827.7 175568.19     0
## 2  232.6035386 195276.8  27357.57  222634.3 167919.19     1
## 3  211.9396813 186285.3  26304.59  212589.9 159980.72     2
## 4  193.1115442 178878.2  25319.42  204197.6 153558.75     2
## 5  175.9560468 172230.4  24489.52  196720.0 147740.91     3
## 6  160.3245966 165269.0  23606.11  188875.1 141662.87     4
## 7  146.0818013 158556.1  22730.75  181286.9 135825.38     4
## 8  133.1042967 152762.9  22006.40  174769.3 130756.48     4
## 9  121.2796778 147861.6  21401.83  169263.4 126459.76     4
## 10 110.5055255 143835.8  20915.24  164751.0 122920.55     4
## 11 100.6885192 140509.2  20544.77  161054.0 119964.42     5
## 12  91.7436287 137695.0  20249.67  157944.6 117445.29     5
## 13  83.5933775 135214.5  19921.63  155136.2 115292.92     5
## 14  76.1671723 132984.1  19592.22  152576.3 113391.86     5
## 15  69.4006906 130786.7  19321.17  150107.9 111465.58     6
## 16  63.2353245 128556.7  19082.31  147639.0 109474.35     6
## 17  57.6176726 126443.0  18834.02  145277.0 107608.99     6
## 18  52.4990774 124598.5  18620.13  143218.6 105978.37     6
## 19  47.8352040 123073.2  18456.25  141529.4 104616.94     6
## 20  43.5856563 121809.4  18331.31  140140.8 103478.13     6
## 21  39.7136268 120771.6  18236.78  139008.4 102534.83     6
## 22  36.1855776 119918.4  18167.18  138085.6 101751.25     6
## 23  32.9709506 119214.7  18118.00  137332.7 101096.70     6
## 24  30.0419022 118635.6  18083.08  136718.6 100552.49     6
## 25  27.3730624 118164.1  18058.55  136222.7 100105.57     6
## 26  24.9413150 117821.8  18043.06  135864.9  99778.74     6
## 27  22.7255973 117569.2  18031.73  135601.0  99537.50     6
## 28  20.7067179 117421.1  18021.94  135443.1  99399.20     6
## 29  18.8671902 117359.9  18023.92  135383.9  99336.02     6
## 30  17.1910810 117354.0  18033.28  135387.3  99320.68     7
## 31  15.6638727 117357.9  18043.29  135401.2  99314.62     7
## 32  14.2723374 117375.3  18055.00  135430.3  99320.31     7
## 33  13.0044223 117405.7  18063.49  135469.2  99342.24     9
## 34  11.8491453 117415.7  18078.12  135493.8  99337.54     9
## 35  10.7964999 117536.1  18072.61  135608.7  99463.51     9
## 36   9.8373686 117723.2  18046.83  135770.0  99676.39     9
## 37   8.9634439 118060.2  17988.83  136049.1 100071.40     9
## 38   8.1671562 118381.0  17925.12  136306.2 100455.91    11
## 39   7.4416086 118401.1  17822.70  136223.8 100578.43    11
## 40   6.7805166 118119.6  17593.99  135713.6 100525.58    12
## 41   6.1781542 117724.2  17346.31  135070.5 100377.88    12
## 42   5.6293040 117312.7  17122.02  134434.8 100190.72    13
## 43   5.1292121 116729.9  16901.88  133631.8  99828.01    13
## 44   4.6735471 116066.9  16699.17  132766.1  99367.77    13
## 45   4.2583620 115522.0  16514.95  132037.0  99007.06    13
## 46   3.8800609 115113.2  16356.46  131469.7  98756.73    13
## 47   3.5353670 114791.4  16214.93  131006.3  98576.44    13
## 48   3.2212947 114546.9  16081.32  130628.2  98465.54    13
## 49   2.9351238 114403.7  15959.12  130362.8  98444.54    13
## 50   2.6743755 114323.0  15850.14  130173.2  98472.90    13
## 51   2.4367913 114270.8  15737.73  130008.6  98533.11    13
## 52   2.2203135 114297.0  15609.21  129906.2  98687.77    14
## 53   2.0230670 114351.2  15477.74  129828.9  98873.42    15
## 54   1.8433433 114376.0  15338.16  129714.1  99037.79    15
## 55   1.6795857 114423.1  15211.18  129634.3  99211.97    17
## 56   1.5303760 114485.9  15101.72  129587.6  99384.20    17
## 57   1.3944216 114591.2  15005.27  129596.4  99585.89    17
## 58   1.2705450 114674.9  14917.58  129592.5  99757.34    17
## 59   1.1576733 114723.7  14834.17  129557.9  99889.53    17
## 60   1.0548288 114842.6  14753.01  129595.6 100089.57    17
## 61   0.9611207 115031.1  14675.47  129706.6 100355.62    17
## 62   0.8757374 115239.8  14603.81  129843.6 100636.00    17
## 63   0.7979393 115450.4  14537.66  129988.1 100912.78    17
## 64   0.7270526 115651.7  14478.17  130129.9 101173.54    17
## 65   0.6624632 115841.1  14419.41  130260.5 101421.72    18
## 66   0.6036118 115985.8  14365.26  130351.0 101620.50    18
## 67   0.5499886 116135.7  14333.37  130469.1 101802.33    18
## 68   0.5011291 116233.9  14290.81  130524.7 101943.08    17
## 69   0.4566102 116360.9  14254.65  130615.6 102106.27    18
## 70   0.4160462 116464.3  14221.60  130685.9 102242.69    18
## 71   0.3790858 116589.3  14192.19  130781.5 102397.08    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   63.23532
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
