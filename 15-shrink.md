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
## (Intercept) 185.946731847
## AtBat         0.096634022
## Hits          0.408580478
## HmRun         1.242303539
## Runs          0.650047295
## RBI           0.642033635
## Walks         0.848737422
## Years         2.608433226
## CAtBat        0.008188531
## CHits         0.031829975
## CHmRun        0.235663247
## CRuns         0.063816873
## CRBI          0.066045116
## CWalks        0.062642350
## LeagueN       4.252099497
## DivisionW   -25.296959330
## PutOuts       0.059902888
## Assists       0.008305300
## Errors       -0.185603402
## NewLeagueN    3.676189338
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
## (Intercept) 185.946731847
## AtBat         0.096634022
## Hits          0.408580478
## HmRun         1.242303539
## Runs          0.650047295
## RBI           0.642033635
## Walks         0.848737422
## Years         2.608433226
## CAtBat        0.008188531
## CHits         0.031829975
## CHmRun        0.235663247
## CRuns         0.063816873
## CRBI          0.066045116
## CWalks        0.062642350
## LeagueN       4.252099497
## DivisionW   -25.296959330
## PutOuts       0.059902888
## Assists       0.008305300
## Errors       -0.185603402
## NewLeagueN    3.676189338
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 681.7166
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 128551
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 452.5069 450.6429 449.9741 449.7012 449.4029 449.0769 448.7208
##  [8] 448.3320 447.9077 447.4450 446.9407 446.3913 445.7932 445.1428
## [15] 444.4358 443.6681 442.8355 441.9334 440.9571 439.9019 438.7631
## [22] 437.5358 436.2154 434.7973 433.2773 431.6515 429.9163 428.0690
## [29] 426.1073 424.0302 421.8372 419.5295 417.1094 414.5805 411.9482
## [36] 409.2195 406.4032 403.5094 400.5500 397.5388 394.4907 391.4216
## [43] 388.3487 385.2893 382.2611 379.2814 376.3671 373.5337 370.7958
## [50] 368.1657 365.6534 363.2680 361.0162 358.9019 356.9269 355.0914
## [57] 353.3935 351.8298 350.3955 349.0849 347.8914 346.8079 345.8270
## [64] 344.9406 344.1426 343.4280 342.7866 342.2149 341.7016 341.2451
## [71] 340.8408 340.4790 340.1563 339.8741 339.6195 339.3897 339.1893
## [78] 339.0040 338.8403 338.6862 338.5430 338.4086 338.2841 338.1570
## [85] 338.0358 337.9179 337.7996 337.6719 337.5497 337.4062 337.2980
## [92] 337.1342 337.0260 336.8539 336.7424 336.5629 336.4531 336.2709
## [99] 336.1633
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.1633
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 363.268
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
##  [1] 449.6027 442.7358 432.9721 424.4699 416.3203 407.2065 398.8628
##  [8] 391.4217 384.9766 379.5421 374.9627 370.8818 367.0994 363.6971
## [15] 360.4987 357.2563 354.4343 351.9885 349.7771 347.9422 346.4463
## [22] 345.2074 344.1903 343.3518 342.7154 342.2631 341.9554 341.7303
## [29] 341.5828 341.4868 341.4385 341.4375 341.4222 341.3896 341.3526
## [36] 341.4302 341.4980 341.4793 341.2994 340.3191 339.2307 338.0187
## [43] 336.9533 336.0770 335.4496 334.9380 334.5364 334.2074 334.0773
## [50] 334.0616 334.0584 334.1093 334.2265 334.3523 334.4464 334.4002
## [57] 334.4097 334.4359 334.4837 334.5159 334.5601 334.6326 334.7193
## [64] 334.7947 334.9056 335.0321 335.1593 335.3053 335.4226 335.5602
## [71] 335.6702 335.7967 335.8575 335.9584 336.0216
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 334.0584
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 360.4987
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
## 1  255.2820965 202142.6  28965.00  231107.6 173177.60     0
## 2  232.6035386 196015.0  29597.30  225612.3 166417.72     1
## 3  211.9396813 187464.9  28227.22  215692.1 159237.63     2
## 4  193.1115442 180174.7  26941.17  207115.9 153233.55     2
## 5  175.9560468 173322.6  25772.75  199095.4 147549.86     3
## 6  160.3245966 165817.2  24552.10  190369.3 141265.05     4
## 7  146.0818013 159091.5  23453.95  182545.5 135637.58     4
## 8  133.1042967 153211.0  22575.37  175786.3 130635.59     4
## 9  121.2796778 148207.0  21916.52  170123.5 126290.45     4
## 10 110.5055255 144052.2  21445.73  165497.9 122606.48     4
## 11 100.6885192 140597.0  21126.92  161723.9 119470.10     5
## 12  91.7436287 137553.3  20959.07  158512.4 116594.27     5
## 13  83.5933775 134762.0  20816.80  155578.8 113945.17     5
## 14  76.1671723 132275.6  20702.41  152978.0 111573.17     5
## 15  69.4006906 129959.3  20680.87  150640.2 109278.47     6
## 16  63.2353245 127632.1  20653.11  148285.2 106978.95     6
## 17  57.6176726 125623.7  20656.32  146280.0 104967.39     6
## 18  52.4990774 123895.9  20654.82  144550.7 103241.07     6
## 19  47.8352040 122344.0  20594.64  142938.7 101749.41     6
## 20  43.5856563 121063.8  20564.82  141628.6 100498.93     6
## 21  39.7136268 120025.0  20559.36  140584.4  99465.67     6
## 22  36.1855776 119168.2  20568.56  139736.7  98599.59     6
## 23  32.9709506 118466.9  20587.54  139054.5  97879.41     6
## 24  30.0419022 117890.5  20612.77  138503.2  97277.70     6
## 25  27.3730624 117453.8  20637.73  138091.6  96816.10     6
## 26  24.9413150 117144.0  20667.28  137811.3  96476.77     6
## 27  22.7255973 116933.5  20705.36  137638.8  96228.13     6
## 28  20.7067179 116779.6  20743.73  137523.3  96035.84     6
## 29  18.8671902 116678.8  20780.42  137459.2  95898.39     6
## 30  17.1910810 116613.2  20815.36  137428.6  95797.87     7
## 31  15.6638727 116580.2  20849.67  137429.9  95730.57     7
## 32  14.2723374 116579.6  20882.26  137461.8  95697.32     7
## 33  13.0044223 116569.1  20911.84  137481.0  95657.28     9
## 34  11.8491453 116546.9  20930.54  137477.4  95616.34     9
## 35  10.7964999 116521.6  20934.17  137455.7  95587.40     9
## 36   9.8373686 116574.6  20923.55  137498.1  95651.04     9
## 37   8.9634439 116620.9  20925.00  137545.9  95695.89     9
## 38   8.1671562 116608.1  20923.75  137531.8  95684.34    11
## 39   7.4416086 116485.3  20888.91  137374.2  95596.39    11
## 40   6.7805166 115817.1  20836.09  136653.2  94980.99    12
## 41   6.1781542 115077.5  20720.38  135797.8  94357.09    12
## 42   5.6293040 114256.6  20600.55  134857.2  93656.07    13
## 43   5.1292121 113537.5  20502.21  134039.7  93035.31    13
## 44   4.6735471 112947.7  20415.96  133363.7  92531.78    13
## 45   4.2583620 112526.5  20351.22  132877.7  92175.24    13
## 46   3.8800609 112183.4  20302.35  132485.8  91881.08    13
## 47   3.5353670 111914.6  20252.38  132167.0  91662.25    13
## 48   3.2212947 111694.6  20196.74  131891.3  91497.84    13
## 49   2.9351238 111607.6  20137.95  131745.6  91469.68    13
## 50   2.6743755 111597.2  20072.83  131670.0  91524.32    13
## 51   2.4367913 111595.0  19997.03  131592.1  91598.02    13
## 52   2.2203135 111629.0  19927.49  131556.5  91701.54    14
## 53   2.0230670 111707.3  19863.20  131570.5  91844.13    15
## 54   1.8433433 111791.5  19810.32  131601.8  91981.16    15
## 55   1.6795857 111854.4  19772.89  131627.3  92081.53    17
## 56   1.5303760 111823.5  19744.55  131568.0  92078.92    17
## 57   1.3944216 111829.9  19750.98  131580.9  92078.89    17
## 58   1.2705450 111847.4  19770.60  131618.0  92076.76    17
## 59   1.1576733 111879.4  19800.35  131679.7  92079.02    17
## 60   1.0548288 111900.9  19833.29  131734.2  92067.57    17
## 61   0.9611207 111930.5  19867.01  131797.5  92063.46    17
## 62   0.8757374 111979.0  19899.27  131878.2  92079.69    17
## 63   0.7979393 112037.0  19930.99  131968.0  92106.03    17
## 64   0.7270526 112087.5  19962.48  132050.0  92125.03    17
## 65   0.6624632 112161.8  19991.16  132152.9  92170.59    18
## 66   0.6036118 112246.5  20018.27  132264.8  92228.26    18
## 67   0.5499886 112331.7  20042.78  132374.5  92288.97    18
## 68   0.5011291 112429.7  20062.16  132491.8  92367.52    17
## 69   0.4566102 112508.3  20083.75  132592.1  92424.55    18
## 70   0.4160462 112600.6  20100.47  132701.1  92500.18    18
## 71   0.3790858 112674.5  20118.49  132793.0  92556.01    18
## 72   0.3454089 112759.4  20135.67  132895.1  92623.78    18
## 73   0.3147237 112800.2  20147.48  132947.7  92652.76    18
## 74   0.2867645 112868.1  20160.67  133028.7  92707.39    18
## 75   0.2612891 112910.5  20170.95  133081.5  92739.57    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   69.40069
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
