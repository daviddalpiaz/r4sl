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
##  [1] 452.3757 450.9546 449.9327 449.6652 449.3728 449.0533 448.7043
##  [8] 448.3233 447.9075 447.4541 446.9599 446.4216 445.8357 445.1985
## [15] 444.5060 443.7541 442.9386 442.0552 441.0992 440.0661 438.9513
## [22] 437.7501 436.4579 435.0703 433.5834 431.9931 430.2963 428.4903
## [29] 426.5729 424.5432 422.4009 420.1472 417.7843 415.3160 412.7477
## [36] 410.0864 407.3405 404.5200 401.6368 398.7041 395.7366 392.7499
## [43] 389.7604 386.7851 383.8409 380.9447 378.1125 375.3594 372.6991
## [50] 370.1432 367.7015 365.3819 363.1907 361.1317 359.2062 357.4140
## [57] 355.7530 354.2198 352.8095 351.5164 350.3341 349.2556 348.2735
## [64] 347.3803 346.5717 345.8419 345.1783 344.5769 344.0307 343.5384
## [71] 343.0945 342.6933 342.3200 341.9861 341.6811 341.3910 341.1274
## [78] 340.8786 340.6422 340.4168 340.1981 339.9842 339.7721 339.5602
## [85] 339.3449 339.1275 338.9082 338.6813 338.4462 338.2060 337.9594
## [92] 337.7057 337.4463 337.1810 336.9100 336.6360 336.3584 336.0797
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.0797
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.1432
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
##  [1] 450.1730 441.7881 432.0133 423.9508 416.5839 408.7234 401.3423
##  [8] 395.0198 389.5108 384.2182 379.5998 375.5067 372.0039 368.6603
## [15] 365.5714 362.5136 359.6963 357.2963 355.3133 353.6767 352.3286
## [22] 351.2207 350.3207 349.6857 349.1927 348.8076 348.5654 348.4524
## [29] 348.5347 348.6749 348.8454 349.0228 349.1885 349.3742 349.6035
## [36] 349.9753 350.5715 351.0396 351.2430 350.9466 349.6387 348.3899
## [43] 347.0932 346.0584 345.1637 344.4311 343.7625 343.1973 342.7431
## [50] 342.4293 342.4736 342.8033 343.2567 343.6733 344.0514 344.4621
## [57] 344.8724 345.2371 345.5711 345.9004 346.1999 346.4677 346.6845
## [64] 346.8659 347.0172 347.1974 347.3586 347.5132 347.6866 347.8110
## [71] 347.9693 348.0678
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.4293
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 375.5067
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
## 1  255.2820965 202655.7  33124.27  235780.0 169531.47     0
## 2  232.6035386 195176.8  33367.16  228543.9 161809.60     1
## 3  211.9396813 186635.4  32022.36  218657.8 154613.08     2
## 4  193.1115442 179734.3  30940.28  210674.6 148793.99     2
## 5  175.9560468 173542.2  30199.54  203741.7 143342.64     3
## 6  160.3245966 167054.8  29700.89  196755.7 137353.92     4
## 7  146.0818013 161075.6  29237.09  190312.7 131838.54     4
## 8  133.1042967 156040.7  28890.20  184930.9 127150.48     4
## 9  121.2796778 151718.7  28582.28  180301.0 123136.42     4
## 10 110.5055255 147623.6  28091.64  175715.2 119531.96     4
## 11 100.6885192 144096.0  27668.34  171764.3 116427.64     5
## 12  91.7436287 141005.3  27347.99  168353.2 113657.26     5
## 13  83.5933775 138386.9  27137.80  165524.7 111249.12     5
## 14  76.1671723 135910.4  26935.01  162845.4 108975.40     5
## 15  69.4006906 133642.4  26812.40  160454.8 106830.04     6
## 16  63.2353245 131416.1  26710.59  158126.7 104705.52     6
## 17  57.6176726 129381.4  26596.64  155978.0 102784.76     6
## 18  52.4990774 127660.6  26498.37  154159.0 101162.26     6
## 19  47.8352040 126247.6  26437.11  152684.7  99810.46     6
## 20  43.5856563 125087.2  26403.25  151490.5  98684.00     6
## 21  39.7136268 124135.4  26391.15  150526.6  97744.27     6
## 22  36.1855776 123356.0  26394.81  149750.8  96961.14     6
## 23  32.9709506 122724.6  26411.43  149136.0  96313.15     6
## 24  30.0419022 122280.1  26451.98  148732.1  95828.12     6
## 25  27.3730624 121935.5  26497.81  148433.3  95437.71     6
## 26  24.9413150 121666.7  26543.33  148210.1  95123.39     6
## 27  22.7255973 121497.8  26579.88  148077.7  94917.97     6
## 28  20.7067179 121419.1  26616.95  148036.0  94802.14     6
## 29  18.8671902 121476.4  26669.31  148145.7  94807.11     6
## 30  17.1910810 121574.2  26730.71  148304.9  94843.45     7
## 31  15.6638727 121693.1  26793.29  148486.4  94899.82     7
## 32  14.2723374 121816.9  26854.12  148671.1  94962.81     7
## 33  13.0044223 121932.6  26915.88  148848.5  95016.75     9
## 34  11.8491453 122062.3  26977.14  149039.5  95085.21     9
## 35  10.7964999 122222.6  27042.22  149264.8  95180.36     9
## 36   9.8373686 122482.7  27100.80  149583.5  95381.89     9
## 37   8.9634439 122900.4  27150.69  150051.1  95749.68     9
## 38   8.1671562 123228.8  27206.88  150435.7  96021.89    11
## 39   7.4416086 123371.7  27247.07  150618.7  96124.60    11
## 40   6.7805166 123163.5  27194.17  150357.7  95969.32    12
## 41   6.1781542 122247.2  26851.48  149098.7  95395.71    12
## 42   5.6293040 121375.5  26487.47  147863.0  94888.06    13
## 43   5.1292121 120473.7  26223.38  146697.0  94250.29    13
## 44   4.6735471 119756.4  25997.65  145754.1  93758.79    13
## 45   4.2583620 119138.0  25797.61  144935.6  93340.39    13
## 46   3.8800609 118632.8  25620.95  144253.7  93011.84    13
## 47   3.5353670 118172.7  25438.87  143611.5  92733.80    13
## 48   3.2212947 117784.4  25270.43  143054.8  92513.94    13
## 49   2.9351238 117472.8  25118.46  142591.3  92354.39    13
## 50   2.6743755 117257.8  24998.52  142256.3  92259.29    13
## 51   2.4367913 117288.2  25000.59  142288.8  92287.60    13
## 52   2.2203135 117514.1  25084.09  142598.2  92429.99    14
## 53   2.0230670 117825.2  25188.92  143014.1  92636.24    15
## 54   1.8433433 118111.3  25287.03  143398.4  92824.30    15
## 55   1.6795857 118371.4  25397.32  143768.7  92974.06    17
## 56   1.5303760 118654.2  25537.90  144192.1  93116.27    17
## 57   1.3944216 118937.0  25679.28  144616.2  93257.68    17
## 58   1.2705450 119188.6  25815.33  145004.0  93373.29    17
## 59   1.1576733 119419.4  25952.97  145372.4  93466.41    17
## 60   1.0548288 119647.1  26080.25  145727.3  93566.82    17
## 61   0.9611207 119854.4  26202.27  146056.7  93652.11    17
## 62   0.8757374 120039.8  26319.00  146358.8  93720.84    17
## 63   0.7979393 120190.1  26423.54  146613.6  93766.57    17
## 64   0.7270526 120316.0  26512.70  146828.7  93803.27    17
## 65   0.6624632 120421.0  26590.09  147011.0  93830.87    18
## 66   0.6036118 120546.0  26663.51  147209.5  93882.50    18
## 67   0.5499886 120658.0  26730.90  147388.9  93927.11    18
## 68   0.5011291 120765.4  26795.75  147561.2  93969.69    17
## 69   0.4566102 120886.0  26864.70  147750.7  94021.25    18
## 70   0.4160462 120972.5  26901.86  147874.4  94070.65    18
## 71   0.3790858 121082.6  26952.91  148035.5  94129.72    18
## 72   0.3454089 121151.2  26989.42  148140.6  94161.80    18
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
