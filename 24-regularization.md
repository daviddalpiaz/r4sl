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
## (Intercept) 135.070668357
## AtBat         0.107381506
## Hits          0.485804001
## HmRun         1.316978601
## Runs          0.754119187
## RBI           0.728063748
## Walks         1.003524589
## Years         2.785904101
## CAtBat        0.009262397
## CHits         0.036847056
## CHmRun        0.271010647
## CRuns         0.073844717
## CRBI          0.076561001
## CWalks        0.068988790
## LeagueN       6.708052750
## DivisionW   -33.295635138
## PutOuts       0.076376801
## Assists       0.010174962
## Errors       -0.297742861
## NewLeagueN    5.296302685
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
## (Intercept) 135.070668357
## AtBat         0.107381506
## Hits          0.485804001
## HmRun         1.316978601
## Runs          0.754119187
## RBI           0.728063748
## Walks         1.003524589
## Years         2.785904101
## CAtBat        0.009262397
## CHits         0.036847056
## CHmRun        0.271010647
## CRuns         0.073844717
## CRBI          0.076561001
## CWalks        0.068988790
## LeagueN       6.708052750
## DivisionW   -33.295635138
## PutOuts       0.076376801
## Assists       0.010174962
## Errors       -0.297742861
## NewLeagueN    5.296302685
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 1193.683
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 122129.1
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.0876 449.3890 448.6356 448.3664 448.0722 447.7507 447.3996
##  [8] 447.0163 446.5981 446.1420 445.6448 445.1034 444.5141 443.8732
## [15] 443.1768 442.4207 441.6008 440.7127 439.7518 438.7135 437.5931
## [22] 436.3861 435.0880 433.6943 432.2011 430.6045 428.9014 427.0892
## [29] 425.1658 423.1305 420.9831 418.7250 416.3586 413.8880 411.3187
## [36] 408.6580 405.9148 403.0993 400.2237 397.3015 394.3478 391.3785
## [43] 388.4103 385.4604 382.5461 379.6843 376.8912 374.1819 371.5701
## [50] 369.0672 366.6833 364.4260 362.3007 360.3113 358.4588 356.7425
## [57] 355.1599 353.7071 352.3789 351.1691 350.0709 349.0769 348.1798
## [64] 347.3708 346.6457 345.9988 345.4178 344.8978 344.4310 344.0164
## [71] 343.6497 343.3178 343.0200 342.7557 342.5144 342.2949 342.0937
## [78] 341.9070 341.7303 341.5623 341.3985 341.2371 341.0766 340.9171
## [85] 340.7518 340.5834 340.4111 340.2302 340.0432 339.8495 339.6474
## [92] 339.4406 339.2278 339.0100 338.7879 338.5636 338.3373 338.1101
## [99] 337.8844
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.8844
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 356.7425
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
##  [1] 450.9172 441.7088 431.1961 422.1452 413.5967 405.3059 397.6662
##  [8] 390.7349 384.7548 379.9891 375.9321 372.3282 369.3145 366.5461
## [15] 363.8722 361.4054 359.1393 357.0003 355.0535 353.3991 352.0420
## [22] 350.9388 350.0488 349.3276 348.7391 348.2577 347.8652 347.5469
## [29] 347.3303 347.2029 347.1469 347.1193 347.0897 347.1257 347.2709
## [36] 347.5335 347.9938 348.5336 348.5668 348.0700 347.4669 346.7579
## [43] 345.9685 345.0670 344.2958 343.6725 343.1926 342.8356 342.6000
## [50] 342.4484 342.6402 342.9310 343.2274 343.5044 343.7596 343.9912
## [57] 344.1560 344.2374 344.3393 344.4323 344.4969 344.5893 344.6995
## [64] 344.8160 344.9318 345.0325 345.1126 345.1539 345.2251 345.2904
## [71] 345.3500 345.3986 345.4113 345.4773 345.5139 345.5381
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.4484
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.3145
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
## 1  255.2820965 203326.3  28881.66  232208.0 174444.66     0
## 2  232.6035386 195106.6  28673.63  223780.3 166433.00     1
## 3  211.9396813 185930.1  27643.88  213573.9 158286.17     2
## 4  193.1115442 178206.6  26725.70  204932.3 151480.85     2
## 5  175.9560468 171062.2  26014.46  197076.7 145047.79     3
## 6  160.3245966 164272.9  25551.38  189824.3 138721.51     4
## 7  146.0818013 158138.4  25149.80  183288.2 132988.65     4
## 8  133.1042967 152673.8  24747.57  177421.3 127926.21     4
## 9  121.2796778 148036.2  24400.18  172436.4 123636.04     4
## 10 110.5055255 144391.7  24130.51  168522.2 120261.18     4
## 11 100.6885192 141325.0  23938.91  165263.9 117386.06     5
## 12  91.7436287 138628.3  23761.04  162389.3 114867.26     5
## 13  83.5933775 136393.2  23624.73  160017.9 112768.44     5
## 14  76.1671723 134356.0  23573.77  157929.8 110782.24     5
## 15  69.4006906 132402.9  23561.51  155964.5 108841.44     6
## 16  63.2353245 130613.9  23559.83  154173.7 107054.03     6
## 17  57.6176726 128981.0  23511.46  152492.5 105469.59     6
## 18  52.4990774 127449.2  23394.89  150844.1 104054.33     6
## 19  47.8352040 126063.0  23220.31  149283.3 102842.70     6
## 20  43.5856563 124890.9  23051.19  147942.1 101839.73     6
## 21  39.7136268 123933.6  22911.84  146845.4 101021.72     6
## 22  36.1855776 123158.0  22797.90  145955.9 100360.14     6
## 23  32.9709506 122534.2  22703.84  145238.0  99830.32     6
## 24  30.0419022 122029.7  22624.97  144654.7  99404.78     6
## 25  27.3730624 121618.9  22558.58  144177.5  99060.36     6
## 26  24.9413150 121283.4  22501.52  143784.9  98781.90     6
## 27  22.7255973 121010.2  22453.05  143463.2  98557.15     6
## 28  20.7067179 120788.8  22411.73  143200.6  98377.10     6
## 29  18.8671902 120638.3  22372.66  143011.0  98265.67     6
## 30  17.1910810 120549.8  22336.63  142886.5  98213.21     7
## 31  15.6638727 120511.0  22297.57  142808.6  98213.40     7
## 32  14.2723374 120491.8  22260.96  142752.7  98230.82     7
## 33  13.0044223 120471.2  22229.52  142700.8  98241.72     9
## 34  11.8491453 120496.2  22190.06  142686.3  98306.16     9
## 35  10.7964999 120597.1  22135.81  142732.9  98461.26     9
## 36   9.8373686 120779.5  22059.62  142839.1  98719.91     9
## 37   8.9634439 121099.7  21952.08  143051.8  99147.62     9
## 38   8.1671562 121475.7  21821.12  143296.8  99654.55    11
## 39   7.4416086 121498.8  21604.42  143103.2  99894.39    11
## 40   6.7805166 121152.7  21332.97  142485.7  99819.74    12
## 41   6.1781542 120733.3  21103.48  141836.8  99629.80    12
## 42   5.6293040 120241.0  20916.42  141157.4  99324.61    13
## 43   5.1292121 119694.2  20746.73  140441.0  98947.50    13
## 44   4.6735471 119071.2  20644.21  139715.4  98427.00    13
## 45   4.2583620 118539.6  20576.15  139115.8  97963.47    13
## 46   3.8800609 118110.8  20500.12  138610.9  97610.65    13
## 47   3.5353670 117781.1  20411.63  138192.8  97369.52    13
## 48   3.2212947 117536.3  20313.48  137849.7  97222.80    13
## 49   2.9351238 117374.7  20223.63  137598.4  97151.12    13
## 50   2.6743755 117270.9  20148.34  137419.2  97122.54    13
## 51   2.4367913 117402.3  20101.67  137504.0  97300.62    13
## 52   2.2203135 117601.7  20075.53  137677.2  97526.16    14
## 53   2.0230670 117805.1  20050.79  137855.9  97754.29    15
## 54   1.8433433 117995.3  20033.62  138028.9  97961.63    15
## 55   1.6795857 118170.7  20024.48  138195.2  98146.19    17
## 56   1.5303760 118330.0  20028.61  138358.6  98301.36    17
## 57   1.3944216 118443.4  20035.16  138478.5  98408.20    17
## 58   1.2705450 118499.4  20030.98  138530.4  98468.41    17
## 59   1.1576733 118569.6  20036.54  138606.1  98533.01    17
## 60   1.0548288 118633.6  20048.40  138682.0  98585.22    17
## 61   0.9611207 118678.1  20056.05  138734.2  98622.08    17
## 62   0.8757374 118741.8  20066.82  138808.6  98675.00    17
## 63   0.7979393 118817.8  20077.44  138895.2  98740.34    17
## 64   0.7270526 118898.0  20088.72  138986.8  98809.33    17
## 65   0.6624632 118978.0  20100.22  139078.2  98877.74    18
## 66   0.6036118 119047.4  20110.63  139158.1  98936.80    18
## 67   0.5499886 119102.7  20124.03  139226.7  98978.65    18
## 68   0.5011291 119131.2  20133.40  139264.6  98997.79    17
## 69   0.4566102 119180.4  20157.42  139337.8  99022.98    18
## 70   0.4160462 119225.5  20155.14  139380.6  99070.35    18
## 71   0.3790858 119266.6  20173.79  139440.4  99092.84    18
## 72   0.3454089 119300.2  20180.03  139480.2  99120.17    18
## 73   0.3147237 119308.9  20190.28  139499.2  99118.67    18
## 74   0.2867645 119354.6  20198.30  139552.9  99156.26    18
## 75   0.2612891 119379.8  20207.36  139587.2  99172.46    18
## 76   0.2380769 119396.6  20219.71  139616.3  99176.90    18
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
