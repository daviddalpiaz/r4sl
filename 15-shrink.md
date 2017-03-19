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
##  [1] 450.9975 449.6151 448.6477 448.3246 448.0351 447.7189 447.3734
##  [8] 446.9963 446.5847 446.1358 445.6466 445.1137 444.5336 443.9027
## [15] 443.2171 442.4726 441.6651 440.7903 439.8436 438.8205 437.7163
## [22] 436.5265 435.2464 433.8718 432.3986 430.8228 429.1413 427.3512
## [29] 425.4505 423.4381 421.3138 419.0785 416.7344 414.2853 411.7361
## [36] 409.0938 406.3669 403.5647 400.6992 397.7832 394.8312 391.8584
## [43] 388.8811 385.9159 382.9796 380.0888 377.2592 374.5058 371.8421
## [50] 369.2800 366.8289 364.4964 362.2897 360.2120 358.2651 356.4490
## [57] 354.7618 353.2003 351.7602 350.4361 349.2219 348.1112 347.0973
## [64] 346.1725 345.3331 344.5717 343.8836 343.2592 342.6925 342.1822
## [71] 341.7259 341.3121 340.9427 340.6098 340.3067 340.0353 339.7972
## [78] 339.5748 339.3725 339.1928 339.0252 338.8693 338.7230 338.5806
## [85] 338.4476 338.3176 338.1867 338.0589 337.9280 337.7950 337.6611
## [92] 337.5227 337.3789 337.2321 337.0830 336.9278 336.7721 336.6129
## [99] 336.4526
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.4526
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 360.212
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
##  [1] 450.9558 442.6164 432.5806 424.2544 416.8130 407.5571 398.6525
##  [8] 391.0088 384.4532 378.9818 374.3919 370.4310 366.8560 363.4794
## [15] 360.1233 357.1127 354.4694 352.1341 350.1189 348.4370 347.0456
## [22] 345.9029 344.9639 344.1859 343.5638 343.1100 342.7945 342.6163
## [29] 342.5717 342.6099 342.6695 342.7525 343.0321 343.4596 343.9421
## [36] 344.6280 345.4376 346.3092 346.8245 346.9757 346.9020 346.0987
## [43] 345.2529 344.3373 343.5108 342.8951 342.4113 342.0129 341.7209
## [50] 341.6261 341.6508 341.9090 342.1702 342.4428 342.6941 342.9898
## [57] 343.3282 343.6457 343.9114 344.1749 344.4553 344.7271 344.9449
## [64] 345.2745 345.5634 345.9102 346.1803 346.4139 346.6539 346.8370
## [71] 347.0570 347.2018 347.3942 347.5347
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.6261
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 366.856
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
## 1  255.2820965 203361.1  23809.87  227171.0 179551.23     0
## 2  232.6035386 195909.3  24007.68  219916.9 171901.59     1
## 3  211.9396813 187125.9  22769.02  209895.0 164356.92     2
## 4  193.1115442 179991.8  21646.95  201638.7 158344.81     2
## 5  175.9560468 173733.1  20761.67  194494.7 152971.40     3
## 6  160.3245966 166102.8  20180.28  186283.1 145922.51     4
## 7  146.0818013 158923.8  19804.25  178728.1 139119.56     4
## 8  133.1042967 152887.8  19633.79  172521.6 133254.06     4
## 9  121.2796778 147804.3  19551.06  167355.3 128253.21     4
## 10 110.5055255 143627.2  19544.46  163171.7 124082.73     4
## 11 100.6885192 140169.3  19615.86  159785.1 120553.41     5
## 12  91.7436287 137219.2  19737.48  156956.6 117481.67     5
## 13  83.5933775 134583.3  19890.70  154474.0 114692.59     5
## 14  76.1671723 132117.3  20086.96  152204.2 112030.31     5
## 15  69.4006906 129688.8  20315.96  150004.7 109372.82     6
## 16  63.2353245 127529.5  20541.42  148070.9 106988.04     6
## 17  57.6176726 125648.6  20712.23  146360.8 104936.35     6
## 18  52.4990774 123998.4  20798.39  144796.8 103200.00     6
## 19  47.8352040 122583.2  20844.89  143428.1 101738.35     6
## 20  43.5856563 121408.4  20895.76  142304.1 100512.61     6
## 21  39.7136268 120440.7  20950.39  141391.1  99490.28     6
## 22  36.1855776 119648.8  21007.65  140656.5  98641.19     6
## 23  32.9709506 119000.1  21064.34  140064.4  97935.77     6
## 24  30.0419022 118463.9  21118.90  139582.8  97345.00     6
## 25  27.3730624 118036.1  21169.26  139205.3  96866.84     6
## 26  24.9413150 117724.4  21206.00  138930.4  96518.45     6
## 27  22.7255973 117508.0  21251.58  138759.6  96256.46     6
## 28  20.7067179 117386.0  21308.66  138694.6  96077.29     6
## 29  18.8671902 117355.4  21370.55  138725.9  95984.84     6
## 30  17.1910810 117381.5  21429.74  138811.3  95951.78     7
## 31  15.6638727 117422.4  21487.29  138909.7  95935.09     7
## 32  14.2723374 117479.3  21539.77  139019.1  95939.51     7
## 33  13.0044223 117671.0  21569.70  139240.8  96101.34     9
## 34  11.8491453 117964.5  21581.59  139546.1  96382.94     9
## 35  10.7964999 118296.2  21584.20  139880.4  96711.97     9
## 36   9.8373686 118768.5  21563.01  140331.5  97205.45     9
## 37   8.9634439 119327.1  21540.46  140867.6  97786.65     9
## 38   8.1671562 119930.1  21515.47  141445.6  98414.63    11
## 39   7.4416086 120287.2  21479.98  141767.2  98807.23    11
## 40   6.7805166 120392.1  21449.89  141842.0  98942.26    12
## 41   6.1781542 120341.0  21280.60  141621.6  99060.37    12
## 42   5.6293040 119784.3  20926.68  140711.0  98857.62    13
## 43   5.1292121 119199.6  20566.27  139765.8  98633.32    13
## 44   4.6735471 118568.2  20224.35  138792.6  98343.85    13
## 45   4.2583620 117999.7  19928.83  137928.5  98070.87    13
## 46   3.8800609 117577.0  19675.47  137252.5  97901.56    13
## 47   3.5353670 117245.5  19454.37  136699.9  97791.14    13
## 48   3.2212947 116972.8  19258.35  136231.2  97714.47    13
## 49   2.9351238 116773.2  19085.40  135858.6  97687.77    13
## 50   2.6743755 116708.4  18936.16  135644.6  97772.24    13
## 51   2.4367913 116725.2  18787.30  135512.5  97937.94    13
## 52   2.2203135 116901.8  18648.42  135550.2  98253.33    14
## 53   2.0230670 117080.5  18526.50  135607.0  98553.97    15
## 54   1.8433433 117267.0  18420.72  135687.8  98846.32    15
## 55   1.6795857 117439.3  18303.92  135743.2  99135.35    17
## 56   1.5303760 117642.0  18172.85  135814.8  99469.13    17
## 57   1.3944216 117874.2  18033.86  135908.1  99840.37    17
## 58   1.2705450 118092.4  17890.83  135983.2 100201.52    17
## 59   1.1576733 118275.1  17756.17  136031.2 100518.90    17
## 60   1.0548288 118456.3  17641.80  136098.1 100814.53    17
## 61   0.9611207 118649.5  17540.16  136189.6 101109.30    17
## 62   0.8757374 118836.8  17447.14  136283.9 101389.65    17
## 63   0.7979393 118987.0  17367.37  136354.3 101619.60    17
## 64   0.7270526 119214.5  17309.28  136523.7 101905.18    17
## 65   0.6624632 119414.1  17262.04  136676.1 102152.03    18
## 66   0.6036118 119653.9  17220.39  136874.3 102433.49    18
## 67   0.5499886 119840.8  17184.67  137025.5 102656.14    18
## 68   0.5011291 120002.6  17156.27  137158.9 102846.34    17
## 69   0.4566102 120168.9  17132.80  137301.7 103036.14    18
## 70   0.4160462 120295.9  17114.83  137410.8 103181.10    18
## 71   0.3790858 120448.5  17094.86  137543.4 103353.67    18
## 72   0.3454089 120549.1  17084.45  137633.5 103464.62    18
## 73   0.3147237 120682.7  17069.53  137752.3 103613.21    18
## 74   0.2867645 120780.4  17061.20  137841.6 103719.17    18
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
