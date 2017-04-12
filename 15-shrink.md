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

![](15-shrink_files/figure-latex/ridge-1.pdf)<!-- --> 

```r
plot(fit_ridge, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/ridge-2.pdf)<!-- --> 

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
##                        1
## (Intercept)  10.35569016
## AtBat         0.04633830
## Hits          0.96376522
## HmRun         0.27163149
## Runs          1.10118079
## RBI           0.87606196
## Walks         1.75331031
## Years         0.50454900
## CAtBat        0.01124891
## CHits         0.06274116
## CHmRun        0.43896753
## CRuns         0.12471202
## CRBI          0.13253839
## CWalks        0.03672947
## LeagueN      25.75710229
## DivisionW   -88.36043520
## PutOuts       0.18483877
## Assists       0.03847012
## Errors       -1.68470904
## NewLeagueN    7.91725602
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 8543.096
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
##  [1] 451.1448 449.8235 448.8545 448.5871 448.2949 447.9756 447.6269
##  [8] 447.2462 446.8308 446.3777 445.8840 445.3462 444.7608 444.1243
## [15] 443.4326 442.6816 441.8672 440.9850 440.0305 438.9991 437.8863
## [22] 436.6873 435.3978 434.0134 432.5301 430.9441 429.2522 427.4518
## [29] 425.5411 423.5190 421.3855 419.1419 416.7907 414.3359 411.7829
## [36] 409.1390 406.4129 403.6149 400.7571 397.8530 394.9175 391.9664
## [43] 389.0166 386.0852 383.1895 380.3466 377.5726 374.8830 372.2919
## [50] 369.8109 367.4498 365.2169 363.1190 361.1598 359.3409 357.6623
## [57] 356.1221 354.7171 353.4426 352.2931 351.2625 350.3440 349.5307
## [64] 348.8109 348.1892 347.6527 347.1945 346.8059 346.4769 346.2152
## [71] 346.0117 345.8581 345.7389 345.6672 345.6345 345.6266 345.6499
## [78] 345.6929 345.7634 345.8428 345.9426 346.0462 346.1600 346.2799
## [85] 346.4003 346.5229 346.6413 346.7560 346.8668 346.9716 347.0674
## [92] 347.1538 347.2316 347.3010 347.3596 347.4082 347.4482 347.4783
## [99] 347.4997
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 345.6266
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.8109
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

![](15-shrink_files/figure-latex/lasso-1.pdf)<!-- --> 

```r
plot(fit_lasso, xvar = "lambda", label = TRUE)
```

![](15-shrink_files/figure-latex/lasso-2.pdf)<!-- --> 

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

![](15-shrink_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

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
##  [1] 448.8614 440.4250 430.1920 421.5063 413.0808 404.3337 396.1035
##  [8] 388.6771 382.3441 377.0400 372.4912 368.4988 365.2307 362.3365
## [15] 359.6582 356.9834 354.3595 352.0661 350.1398 348.5520 347.2456
## [22] 346.1727 345.3057 344.6528 344.2014 343.9242 343.8099 343.7846
## [29] 343.8381 343.9581 344.1219 344.2914 344.4351 344.6508 345.1173
## [36] 345.8472 346.8249 347.6611 348.1058 347.7707 347.2728 346.5399
## [43] 345.5937 344.6797 343.8785 343.2856 342.8536 342.4780 342.1995
## [50] 342.0200 342.0635 342.1646 342.3349 342.5598 342.7759 343.0212
## [57] 343.2563 343.3761 343.4714 343.6025 343.7710 343.9599 344.1262
## [64] 344.2928 344.4736 344.5845 344.7920 344.8850 345.0666 345.1614
## [71] 345.3468
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.02
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 362.3365
```

## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
#fit_lasso_cv
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high conf.low nzero
## 1  255.2820965 201476.5  20999.34  222475.9 180477.2     0
## 2  232.6035386 193974.2  21349.33  215323.5 172624.8     1
## 3  211.9396813 185065.2  20483.81  205549.0 164581.4     2
## 4  193.1115442 177667.6  19688.72  197356.3 157978.8     2
## 5  175.9560468 170635.8  18994.64  189630.4 151641.1     3
## 6  160.3245966 163485.7  18335.20  181821.0 145150.5     4
## 7  146.0818013 156898.0  17706.48  174604.4 139191.5     4
## 8  133.1042967 151069.9  17130.83  168200.8 133939.1     4
## 9  121.2796778 146187.0  16711.14  162898.2 129475.9     4
## 10 110.5055255 142159.2  16422.63  158581.8 125736.6     4
## 11 100.6885192 138749.7  16245.84  154995.6 122503.9     5
## 12  91.7436287 135791.3  16088.96  151880.3 119702.4     5
## 13  83.5933775 133393.4  15990.20  149383.6 117403.2     5
## 14  76.1671723 131287.7  15935.93  147223.7 115351.8     5
## 15  69.4006906 129354.0  15921.27  145275.3 113432.7     6
## 16  63.2353245 127437.2  15939.99  143377.2 111497.2     6
## 17  57.6176726 125570.6  15893.03  141463.7 109677.6     6
## 18  52.4990774 123950.5  15820.25  139770.8 108130.3     6
## 19  47.8352040 122597.9  15777.79  138375.7 106820.1     6
## 20  43.5856563 121488.5  15758.80  137247.3 105729.7     6
## 21  39.7136268 120579.5  15757.57  136337.1 104822.0     6
## 22  36.1855776 119835.6  15770.03  135605.6 104065.5     6
## 23  32.9709506 119236.0  15791.34  135027.4 103444.7     6
## 24  30.0419022 118785.5  15808.81  134594.3 102976.7     6
## 25  27.3730624 118474.6  15819.33  134293.9 102655.3     6
## 26  24.9413150 118283.8  15830.76  134114.6 102453.1     6
## 27  22.7255973 118205.3  15845.78  134051.0 102359.5     6
## 28  20.7067179 118187.9  15865.63  134053.5 102322.2     6
## 29  18.8671902 118224.7  15891.71  134116.4 102333.0     6
## 30  17.1910810 118307.2  15920.59  134227.8 102386.6     7
## 31  15.6638727 118419.9  15947.58  134367.4 102472.3     7
## 32  14.2723374 118536.6  15973.71  134510.3 102562.9     7
## 33  13.0044223 118635.5  15993.37  134628.9 102642.2     9
## 34  11.8491453 118784.2  16006.55  134790.8 102777.7     9
## 35  10.7964999 119105.9  16040.00  135145.9 103065.9     9
## 36   9.8373686 119610.3  16091.36  135701.6 103518.9     9
## 37   8.9634439 120287.5  16108.65  136396.1 104178.8     9
## 38   8.1671562 120868.2  16082.69  136950.9 104785.5    11
## 39   7.4416086 121177.6  15983.39  137161.0 105194.3    11
## 40   6.7805166 120944.4  15830.80  136775.2 105113.6    12
## 41   6.1781542 120598.4  15699.31  136297.7 104899.1    12
## 42   5.6293040 120089.9  15612.25  135702.2 104477.7    13
## 43   5.1292121 119435.0  15530.52  134965.5 103904.4    13
## 44   4.6735471 118804.1  15422.45  134226.5 103381.6    13
## 45   4.2583620 118252.5  15298.07  133550.5 102954.4    13
## 46   3.8800609 117845.0  15157.97  133003.0 102687.0    13
## 47   3.5353670 117548.6  15029.00  132577.6 102519.6    13
## 48   3.2212947 117291.2  14922.73  132213.9 102368.4    13
## 49   2.9351238 117100.5  14822.48  131923.0 102278.0    13
## 50   2.6743755 116977.7  14720.64  131698.3 102257.1    13
## 51   2.4367913 117007.4  14629.44  131636.9 102378.0    13
## 52   2.2203135 117076.6  14542.16  131618.8 102534.5    14
## 53   2.0230670 117193.2  14469.47  131662.7 102723.7    15
## 54   1.8433433 117347.2  14408.89  131756.1 102938.3    15
## 55   1.6795857 117495.3  14342.94  131838.2 103152.4    17
## 56   1.5303760 117663.5  14286.11  131949.6 103377.4    17
## 57   1.3944216 117824.9  14245.25  132070.1 103579.6    17
## 58   1.2705450 117907.2  14230.49  132137.6 103676.7    17
## 59   1.1576733 117972.6  14219.28  132191.9 103753.3    17
## 60   1.0548288 118062.7  14200.49  132263.2 103862.2    17
## 61   0.9611207 118178.5  14182.48  132361.0 103996.0    17
## 62   0.8757374 118308.4  14166.57  132475.0 104141.9    17
## 63   0.7979393 118422.8  14144.22  132567.1 104278.6    17
## 64   0.7270526 118537.6  14123.26  132660.8 104414.3    17
## 65   0.6624632 118662.1  14105.23  132767.3 104556.9    18
## 66   0.6036118 118738.5  14093.02  132831.5 104645.5    18
## 67   0.5499886 118881.5  14076.31  132957.8 104805.2    18
## 68   0.5011291 118945.6  14070.53  133016.2 104875.1    17
## 69   0.4566102 119071.0  14060.85  133131.8 105010.1    18
## 70   0.4160462 119136.4  14058.59  133195.0 105077.8    18
## 71   0.3790858 119264.4  14044.44  133308.8 105219.9    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   76.16717
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

![](15-shrink_files/figure-latex/unnamed-chunk-15-1.pdf)<!-- --> 


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

![](15-shrink_files/figure-latex/unnamed-chunk-19-1.pdf)<!-- --> 


```r
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```

![](15-shrink_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 

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

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.3.3 and the following packages:

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
