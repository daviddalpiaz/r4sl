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
## (Intercept)  1.381737e+01
## AtBat       -4.269835e-02
## Hits         1.207444e+00
## HmRun       -3.713857e-01
## Runs         1.150774e+00
## RBI          8.599947e-01
## Walks        2.015222e+00
## Years       -1.517933e+00
## CAtBat       1.049592e-02
## CHits        7.459755e-02
## CHmRun       5.041878e-01
## CRuns        1.477624e-01
## CRBI         1.572011e-01
## CWalks      -8.252263e-03
## LeagueN      3.270677e+01
## DivisionW   -1.024976e+02
## PutOuts      2.141219e-01
## Assists      6.101918e-02
## Errors      -2.296448e+00
## NewLeagueN   3.503041e+00
```

```r
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2) # penalty term for lambda minimum
```

```
## [1] 11603.43
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
##  [1] 451.4312 449.3826 448.9420 448.6704 448.3736 448.0492 447.6949
##  [8] 447.3081 446.8861 446.4258 445.9241 445.3776 444.7828 444.1358
## [15] 443.4328 442.6694 441.8415 440.9446 439.9741 438.9253 437.7934
## [22] 436.5738 435.2619 433.8532 432.3435 430.7289 429.0062 427.1725
## [29] 425.2258 423.1650 420.9900 418.7019 416.3030 413.7973 411.1901
## [36] 408.4886 405.7015 402.8390 399.9132 396.9378 393.9276 390.8986
## [43] 387.8677 384.8523 381.8698 378.9375 376.0719 373.2886 370.6016
## [50] 368.0229 365.5631 363.2310 361.0324 358.9716 357.0504 355.2688
## [57] 353.6251 352.1161 350.7370 349.4824 348.3459 347.3207 346.3998
## [64] 345.5753 344.8421 344.1950 343.6257 343.1248 342.6875 342.3119
## [71] 341.9901 341.7159 341.4883 341.3009 341.1461 341.0246 340.9297
## [78] 340.8596 340.8127 340.7810 340.7688 340.7622 340.7747 340.7873
## [85] 340.8189 340.8383 340.8860 340.9067 340.9614 340.9853 341.0465
## [92] 341.0698 341.1394 341.1610 341.2378 341.2627 341.3414 341.3792
## [99] 341.4664
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.7622
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 368.0229
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
##  [1] 451.6462 444.0495 433.9638 425.5213 417.9099 410.0463 402.5542
##  [8] 395.9882 390.5987 385.9537 382.0981 378.5232 375.3586 372.7681
## [15] 370.2098 367.2619 364.3818 361.8304 359.7211 357.9557 356.4877
## [22] 355.2720 354.2634 353.4387 352.8417 352.3949 352.1010 351.9215
## [29] 351.8440 351.8465 351.8769 351.9638 352.3964 352.9578 353.3993
## [36] 353.9828 354.5961 355.1548 355.5796 355.5773 355.2880 354.3353
## [43] 353.2787 352.4162 351.7215 351.2295 350.8782 350.6729 350.5409
## [50] 350.4653 350.4966 350.5990 350.7625 351.0218 351.2648 351.5223
## [57] 351.7815 351.9988 352.1532 352.2771 352.2966 352.2806 352.3030
## [64] 352.3433 352.3981 352.4609 352.5160 352.5842 352.6757 352.7720
## [71] 352.8769 352.9883 353.0822 353.1746 353.2640
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 350.4653
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 378.5232
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
## 1  255.2820965 203984.3  26900.08  230884.4 177084.2     0
## 2  232.6035386 197180.0  26897.12  224077.1 170282.8     1
## 3  211.9396813 188324.6  25728.34  214053.0 162596.3     2
## 4  193.1115442 181068.4  24705.53  205773.9 156362.8     2
## 5  175.9560468 174648.7  23952.52  198601.2 150696.2     3
## 6  160.3245966 168138.0  23465.39  191603.4 144672.6     4
## 7  146.0818013 162049.9  23051.06  185100.9 138998.8     4
## 8  133.1042967 156806.6  22725.34  179532.0 134081.3     4
## 9  121.2796778 152567.4  22496.90  175064.3 130070.4     4
## 10 110.5055255 148960.3  22359.03  171319.3 126601.2     4
## 11 100.6885192 145998.9  22322.99  168321.9 123675.9     5
## 12  91.7436287 143279.8  22392.02  165671.8 120887.8     5
## 13  83.5933775 140894.1  22496.12  163390.2 118397.9     5
## 14  76.1671723 138956.0  22619.53  161575.6 116336.5     5
## 15  69.4006906 137055.3  22692.69  159748.0 114362.6     6
## 16  63.2353245 134881.3  22640.17  157521.4 112241.1     6
## 17  57.6176726 132774.1  22547.70  155321.8 110226.4     6
## 18  52.4990774 130921.2  22467.50  153388.7 108453.7     6
## 19  47.8352040 129399.3  22435.62  151834.9 106963.6     6
## 20  43.5856563 128132.3  22430.16  150562.4 105702.1     6
## 21  39.7136268 127083.5  22444.83  149528.3 104638.6     6
## 22  36.1855776 126218.2  22474.22  148692.4 103744.0     6
## 23  32.9709506 125502.6  22514.27  148016.9 102988.3     6
## 24  30.0419022 124918.9  22560.30  147479.2 102358.6     6
## 25  27.3730624 124497.2  22607.86  147105.1 101889.4     6
## 26  24.9413150 124182.2  22655.32  146837.5 101526.8     6
## 27  22.7255973 123975.1  22701.07  146676.2 101274.1     6
## 28  20.7067179 123848.7  22744.17  146592.9 101104.5     6
## 29  18.8671902 123794.2  22784.64  146578.8 101009.6     6
## 30  17.1910810 123795.9  22823.84  146619.8 100972.1     7
## 31  15.6638727 123817.3  22862.69  146680.0 100954.6     7
## 32  14.2723374 123878.5  22916.59  146795.1 100961.9     7
## 33  13.0044223 124183.2  22996.10  147179.3 101187.1     9
## 34  11.8491453 124579.2  23083.88  147663.1 101495.3     9
## 35  10.7964999 124891.0  23166.10  148057.1 101724.9     9
## 36   9.8373686 125303.8  23241.52  148545.3 102062.3     9
## 37   8.9634439 125738.4  23313.67  149052.0 102424.7     9
## 38   8.1671562 126134.9  23439.51  149574.4 102695.4    11
## 39   7.4416086 126436.8  23565.83  150002.7 102871.0    11
## 40   6.7805166 126435.2  23644.60  150079.8 102790.6    12
## 41   6.1781542 126229.6  23697.90  149927.5 102531.7    12
## 42   5.6293040 125553.5  23587.77  149141.3 101965.8    13
## 43   5.1292121 124805.8  23369.01  148174.8 101436.8    13
## 44   4.6735471 124197.2  23190.47  147387.6 101006.7    13
## 45   4.2583620 123708.0  23069.21  146777.2 100638.8    13
## 46   3.8800609 123362.1  23006.64  146368.8 100355.5    13
## 47   3.5353670 123115.5  22965.67  146081.2 100149.8    13
## 48   3.2212947 122971.5  22920.08  145891.6 100051.4    13
## 49   2.9351238 122878.9  22860.98  145739.9 100017.9    13
## 50   2.6743755 122825.9  22799.81  145625.7 100026.1    13
## 51   2.4367913 122847.9  22737.66  145585.6 100110.2    13
## 52   2.2203135 122919.7  22680.12  145599.8 100239.6    14
## 53   2.0230670 123034.3  22644.18  145678.5 100390.2    15
## 54   1.8433433 123216.3  22623.67  145840.0 100592.6    15
## 55   1.6795857 123387.0  22608.32  145995.3 100778.7    17
## 56   1.5303760 123567.9  22595.56  146163.5 100972.3    17
## 57   1.3944216 123750.2  22564.32  146314.5 101185.9    17
## 58   1.2705450 123903.1  22515.04  146418.2 101388.1    17
## 59   1.1576733 124011.9  22459.26  146471.1 101552.6    17
## 60   1.0548288 124099.2  22391.76  146490.9 101707.4    17
## 61   0.9611207 124112.9  22333.43  146446.3 101779.5    17
## 62   0.8757374 124101.6  22282.73  146384.4 101818.9    17
## 63   0.7979393 124117.4  22243.75  146361.2 101873.7    17
## 64   0.7270526 124145.8  22220.50  146366.3 101925.3    17
## 65   0.6624632 124184.4  22209.08  146393.5 101975.4    18
## 66   0.6036118 124228.7  22194.87  146423.5 102033.8    18
## 67   0.5499886 124267.5  22188.18  146455.7 102079.4    18
## 68   0.5011291 124315.6  22179.56  146495.2 102136.1    17
## 69   0.4566102 124380.1  22174.02  146554.2 102206.1    18
## 70   0.4160462 124448.1  22166.31  146614.4 102281.8    18
## 71   0.3790858 124522.1  22156.73  146678.9 102365.4    18
## 72   0.3454089 124600.8  22142.44  146743.2 102458.3    18
## 73   0.3147237 124667.0  22129.77  146796.8 102537.2    18
## 74   0.2867645 124732.3  22116.02  146848.3 102616.2    18
## 75   0.2612891 124795.5  22106.04  146901.5 102689.4    18
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
