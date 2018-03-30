# Regularization

**Chapter Status:** Currently this chapter is very sparse. It essentially only expands upon an example discussed in ISL, thus only illustrates usage of the methods. Mathematical and conceptual details of the methods will be added later. Also, more comments on using `glmnet` with `caret` will be discussed.



We will use the `Hitters` dataset from the `ISLR` package to explore two shrinkage methods: **ridge regression** and **lasso**. These are otherwise known as **penalized regression** methods.


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

We use the `glmnet()` and `cv.glmnet()` functions from the `glmnet` package to fit penalized regressions.


```r
library(glmnet)
```

Unfortunately, the `glmnet` function does not allow the use of model formulas, so we setup the data for ease of use with `glmnet`. Eventually we will use `train()` from `caret` which does allow for fitting penalized regression with the formula syntax, but to explore some of the details, we first work with the functions from `glmnet` directly.


```r
X = model.matrix(Salary ~ ., Hitters)[, -1]
y = Hitters$Salary
```

First, we fit an ordinary linear regression, and note the size of the predictors' coefficients, and predictors' coefficients squared. (The two penalties we will use.)


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

Notice that the intercept is **not** penalized. Also, note that that ridge regression is **not** scale invariant like the usual unpenalized regression. Thankfully, `glmnet()` takes care of this internally. It automatically standardizes predictors for fitting, then reports fitted coefficient using the original scale.

The two plots illustrate how much the coefficients are penalized for different values of $\lambda$. Notice none of the coefficients are forced to be zero.


```r
par(mfrow = c(1, 2))
fit_ridge = glmnet(X, y, alpha = 0)
plot(fit_ridge)
plot(fit_ridge, xvar = "lambda", label = TRUE)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/ridge-1} \end{center}

We use cross-validation to select a good $\lambda$ value. The `cv.glmnet()`function uses 10 folds by default. The plot illustrates the MSE for the $\lambda$s considered. Two lines are drawn. The first is the $\lambda$ that gives the smallest MSE. The second is the $\lambda$ that gives an MSE within one standard error of the smallest.


```r
fit_ridge_cv = cv.glmnet(X, y, alpha = 0)
plot(fit_ridge_cv)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-7-1} \end{center}

The `cv.glmnet()` function returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are smaller than the full linear regression. (As we would expect.)


```r
# fitted coefficients, using 1-SE rule lambda, default behavior
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
# fitted coefficients, using minimum lambda
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
# penalty term using minimum lambda
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 18126.85
```


```r
# fitted coefficients, using 1-SE rule lambda
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
# penalty term using 1-SE rule lambda
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 507.788
```


```r
# predict using minimum lambda
predict(fit_ridge_cv, X, s = "lambda.min")
```


```r
# predict using 1-SE rule lambda, default behavior
predict(fit_ridge_cv, X)
```


```r
# calcualte "train error"
mean((y - predict(fit_ridge_cv, X)) ^ 2)
```

```
## [1] 132355.6
```


```r
# CV-RMSEs
sqrt(fit_ridge_cv$cvm)
```

```
##  [1] 450.9735 449.2528 448.5197 448.2535 447.9626 447.6447 447.2975
##  [8] 446.9184 446.5048 446.0538 445.5622 445.0268 444.4441 443.8103
## [15] 443.1217 442.3740 441.5633 440.6850 439.7348 438.7081 437.6003
## [22] 436.4068 435.1232 433.7452 432.2688 430.6902 429.0063 427.2146
## [29] 425.3131 423.3009 421.1780 418.9458 416.6067 414.1647 411.6255
## [36] 408.9961 406.2856 403.5040 400.6635 397.7777 394.8615 391.9308
## [43] 389.0024 386.0934 383.2211 380.4026 377.6541 374.9910 372.4270
## [50] 369.9740 367.6417 365.4381 363.3699 361.4402 359.6506 358.0007
## [57] 356.4883 355.1097 353.8598 352.7327 351.7215 350.8190 350.0177
## [64] 349.3067 348.6830 348.1457 347.6755 347.2688 346.9145 346.6147
## [71] 346.3678 346.1549 345.9735 345.8275 345.7016 345.5970 345.5063
## [78] 345.4263 345.3544 345.2876 345.2195 345.1503 345.0741 344.9911
## [85] 344.8972 344.7950 344.6790 344.5495 344.4032 344.2438 344.0705
## [92] 343.8833 343.6807 343.4655 343.2382 343.0002 342.7529 342.4988
## [99] 342.2381
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min])
```

```
## [1] 342.2381
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) 
```

```
## [1] 369.974
```


## Lasso

We now illustrate **lasso**, which can be fit using `glmnet()` with `alpha = 1` and seeks to minimize

$$
\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}    \right) ^ 2 + \lambda \sum_{j=1}^{p} |\beta_j| .
$$

Like ridge, lasso is not scale invariant.

The two plots illustrate how much the coefficients are penalized for different values of $\lambda$. Notice some of the coefficients are forced to be zero.


```r
par(mfrow = c(1, 2))
fit_lasso = glmnet(X, y, alpha = 1)
plot(fit_lasso)
plot(fit_lasso, xvar = "lambda", label = TRUE)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/lasso-1} \end{center}

Again, to actually pick a $\lambda$, we will use cross-validation. The plot is similar to the ridge plot. Notice along the top is the number of features in the model. (Which changed in this plot.)


```r
fit_lasso_cv = cv.glmnet(X, y, alpha = 1)
plot(fit_lasso_cv)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-19-1} \end{center}

`cv.glmnet()` returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are again smaller than the full linear regression. (As we would expect.) Some coefficients are 0.


```r
# fitted coefficients, using 1-SE rule lambda, default behavior
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
# fitted coefficients, using minimum lambda
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
# penalty term using minimum lambda
sum(coef(fit_lasso_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 15414.79
```


```r
# fitted coefficients, using 1-SE rule lambda
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
# penalty term using 1-SE rule lambda
sum(coef(fit_lasso_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 3.255212
```


```r
# predict using minimum lambda
predict(fit_lasso_cv, X, s = "lambda.min")
```


```r
# predict using 1-SE rule lambda, default behavior
predict(fit_lasso_cv, X)
```


```r
# calcualte "train error"
mean((y - predict(fit_lasso_cv, X)) ^ 2)
```

```
## [1] 127112.4
```


```r
# CV-RMSEs
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 448.7412 441.3522 431.3419 423.0511 415.5221 407.4414 399.1191
##  [8] 391.8171 385.7155 380.5748 376.0501 372.1215 368.7802 365.7393
## [15] 362.9071 360.0275 357.2858 354.9293 352.9621 351.3226 349.9716
## [22] 348.8700 347.9737 347.2530 346.7122 346.3207 346.0509 345.8565
## [29] 345.7286 345.6722 345.6774 345.6901 345.6351 345.8034 346.0437
## [36] 346.4435 347.0380 347.5599 347.5566 346.7979 345.7785 344.7823
## [43] 343.7602 342.8260 342.0129 341.2692 340.6602 340.3716 340.2748
## [50] 340.1740 340.1873 340.3540 340.7045 341.1260 341.5725 341.9257
## [57] 342.1345 342.2902 342.4581 342.5964 342.7378 342.8480 342.9313
## [64] 343.0283 343.1751 343.3507 343.5257 343.7148 343.8906 344.0445
## [71] 344.1891 344.3231
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min])
```

```
## [1] 340.174
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) 
```

```
## [1] 372.1215
```


## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
# the output from the commented line would be immense
# fit_lasso_cv
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high  conf.low nzero
## 1  255.2820965 201368.7  25382.62  226751.3 175986.07     0
## 2  232.6035386 194791.8  25228.06  220019.8 169563.72     1
## 3  211.9396813 186055.8  24199.85  210255.6 161855.95     2
## 4  193.1115442 178972.2  23368.85  202341.1 155603.37     2
## 5  175.9560468 172658.6  22806.74  195465.4 149851.91     3
## 6  160.3245966 166008.5  22434.57  188443.1 143573.94     4
## 7  146.0818013 159296.1  22196.81  181492.9 137099.28     4
## 8  133.1042967 153520.6  22111.64  175632.3 131408.97     4
## 9  121.2796778 148776.5  22125.88  170902.3 126650.57     4
## 10 110.5055255 144837.2  22203.46  167040.6 122633.70     4
## 11 100.6885192 141413.7  22374.94  163788.6 119038.75     5
## 12  91.7436287 138474.4  22596.80  161071.2 115877.61     5
## 13  83.5933775 135998.8  22821.74  158820.6 113177.07     5
## 14  76.1671723 133765.2  23055.56  156820.8 110709.66     5
## 15  69.4006906 131701.5  23289.90  154991.4 108411.64     6
## 16  63.2353245 129619.8  23449.69  153069.5 106170.14     6
## 17  57.6176726 127653.1  23527.61  151180.7 104125.50     6
## 18  52.4990774 125974.8  23593.59  149568.4 102381.24     6
## 19  47.8352040 124582.2  23671.85  148254.1 100910.38     6
## 20  43.5856563 123427.6  23758.50  147186.1  99669.09     6
## 21  39.7136268 122480.1  23849.07  146329.2  98631.02     6
## 22  36.1855776 121710.3  23939.89  145650.2  97770.40     6
## 23  32.9709506 121085.7  24031.06  145116.7  97054.62     6
## 24  30.0419022 120584.6  24122.94  144707.6  96461.69     6
## 25  27.3730624 120209.4  24218.02  144427.4  95991.34     6
## 26  24.9413150 119938.0  24313.09  144251.1  95624.95     6
## 27  22.7255973 119751.2  24401.41  144152.7  95349.84     6
## 28  20.7067179 119616.7  24476.59  144093.3  95140.12     6
## 29  18.8671902 119528.3  24541.52  144069.8  94986.73     6
## 30  17.1910810 119489.3  24598.09  144087.3  94891.16     7
## 31  15.6638727 119492.8  24652.50  144145.3  94840.35     7
## 32  14.2723374 119501.6  24705.06  144206.7  94796.59     7
## 33  13.0044223 119463.6  24760.16  144223.8  94703.45     9
## 34  11.8491453 119580.0  24800.64  144380.6  94779.36     9
## 35  10.7964999 119746.3  24833.10  144579.4  94913.16     9
## 36   9.8373686 120023.1  24837.40  144860.5  95185.69     9
## 37   8.9634439 120435.4  24817.98  145253.4  95617.42     9
## 38   8.1671562 120797.9  24738.02  145535.9  96059.84    11
## 39   7.4416086 120795.6  24623.86  145419.5  96171.76    11
## 40   6.7805166 120268.8  24545.88  144814.7  95722.92    12
## 41   6.1781542 119562.8  24485.86  144048.6  95076.91    12
## 42   5.6293040 118874.9  24328.80  143203.7  94546.05    13
## 43   5.1292121 118171.1  24146.04  142317.1  94025.05    13
## 44   4.6735471 117529.7  23989.21  141518.9  93540.45    13
## 45   4.2583620 116972.8  23835.53  140808.4  93137.31    13
## 46   3.8800609 116464.7  23662.44  140127.1  92802.22    13
## 47   3.5353670 116049.4  23497.40  139546.8  92552.01    13
## 48   3.2212947 115852.8  23349.26  139202.1  92503.53    13
## 49   2.9351238 115786.9  23210.85  138997.8  92576.08    13
## 50   2.6743755 115718.4  23086.38  138804.7  92631.98    13
## 51   2.4367913 115727.4  23000.47  138727.9  92726.92    13
## 52   2.2203135 115840.8  22943.33  138784.2  92897.49    14
## 53   2.0230670 116079.6  22893.36  138972.9  93186.20    15
## 54   1.8433433 116367.0  22827.22  139194.2  93539.76    15
## 55   1.6795857 116671.8  22745.98  139417.8  93925.82    17
## 56   1.5303760 116913.2  22685.69  139598.9  94227.48    17
## 57   1.3944216 117056.0  22619.43  139675.5  94436.62    17
## 58   1.2705450 117162.6  22538.88  139701.5  94623.71    17
## 59   1.1576733 117277.5  22463.29  139740.8  94814.24    17
## 60   1.0548288 117372.3  22398.57  139770.9  94973.72    17
## 61   0.9611207 117469.2  22328.64  139797.8  95140.56    17
## 62   0.8757374 117544.7  22257.69  139802.4  95287.04    17
## 63   0.7979393 117601.8  22195.54  139797.4  95406.30    17
## 64   0.7270526 117668.4  22141.05  139809.5  95527.36    17
## 65   0.6624632 117769.1  22099.95  139869.1  95669.18    18
## 66   0.6036118 117889.7  22062.99  139952.7  95826.74    18
## 67   0.5499886 118009.9  22033.16  140043.0  95976.72    18
## 68   0.5011291 118139.9  22014.83  140154.7  96125.06    17
## 69   0.4566102 118260.8  22001.92  140262.7  96258.84    18
## 70   0.4160462 118366.6  21993.47  140360.1  96373.11    18
## 71   0.3790858 118466.1  21982.87  140449.0  96483.28    18
## 72   0.3454089 118558.4  21972.29  140530.7  96586.11    18
```

```r
# the two lambda values of interest
glance(fit_lasso_cv) 
```

```
##   lambda.min lambda.1se
## 1   2.674375   91.74363
```


## Simulated Data, $p > n$

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



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-34-1} \end{center}


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
par(mfrow = c(1, 2))
plot(glmnet(X, y, family = "binomial"))
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-38-1} \end{center}

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

The interaction between the `glmnet` and `caret` packages is sometimes frustrating, but for obtaining results for particular values of $\lambda$, we see it can be easily used. More on this next chapter.


## External Links

- [`glmnet` Web Vingette](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html) - Details from the package developers.


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](24-regularization.Rmd). The file was created using `R` version 3.4.4. The following packages (and their dependencies) were loaded when knitting this file:


```
## [1] "caret"   "ggplot2" "lattice" "broom"   "glmnet"  "foreach" "Matrix"
```
