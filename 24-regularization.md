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
##  [1] 451.6467 449.7118 449.1547 448.8854 448.5911 448.2695 447.9182
##  [8] 447.5346 447.1161 446.6597 446.1621 445.6202 445.0302 444.3886
## [15] 443.6913 442.9341 442.1129 441.2231 440.2603 439.2197 438.0966
## [22] 436.8864 435.5844 434.1863 432.6877 431.0850 429.3746 427.5539
## [29] 425.6207 423.5740 421.4135 419.1404 416.7569 414.2668 411.6755
## [36] 408.9901 406.2193 403.3731 400.4636 397.5044 394.5104 391.4976
## [43] 388.4829 385.4836 382.5173 379.6013 376.7523 373.9860 371.3164
## [50] 368.7561 366.3153 364.0029 361.8254 359.7869 357.8893 356.1326
## [57] 354.5153 353.0340 351.6840 350.4598 349.3550 348.3626 347.4755
## [64] 346.6859 345.9908 345.3806 344.8458 344.3820 343.9810 343.6439
## [71] 343.3609 343.1222 342.9263 342.7745 342.6511 342.5584 342.4910
## [78] 342.4435 342.4150 342.3994 342.3933 342.3951 342.4000 342.4069
## [85] 342.4121 342.4148 342.4137 342.4040 342.3877 342.3616 342.3252
## [92] 342.2789 342.2220 342.1533 342.0722 341.9826 341.8803 341.7699
## [99] 341.6499
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.6499
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.3164
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
##  [1] 449.6890 441.6392 431.9333 423.5274 415.5520 407.3068 399.4804
##  [8] 392.3637 386.2222 380.9689 376.3846 372.4711 369.1636 366.3535
## [15] 363.4378 360.2399 357.4296 354.9967 352.9366 351.2305 349.8183
## [22] 348.6526 347.6949 346.9052 346.2714 345.8001 345.4710 345.2559
## [29] 345.1148 345.0386 344.9998 344.9822 344.9684 344.9179 344.7934
## [36] 344.8830 345.2902 345.7789 345.9788 345.8751 345.6219 345.2618
## [43] 344.5251 343.6665 342.9764 342.4560 342.0130 341.6492 341.4003
## [50] 341.3130 341.3863 341.5140 341.6381 341.7715 341.9521 342.2448
## [57] 342.5241 342.7462 342.9910 343.1852 343.2902 343.3717 343.4549
## [64] 343.5426 343.6627 343.7668 343.8825 343.9844 344.1100 344.2444
## [71] 344.3626 344.4584 344.5577 344.6627 344.7522 344.8077
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.313
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 369.1636
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
## 1  255.2820965 202220.2  21645.80  223866.0 180574.41     0
## 2  232.6035386 195045.2  21297.77  216342.9 173747.40     1
## 3  211.9396813 186566.3  20276.90  206843.3 166289.44     2
## 4  193.1115442 179375.5  19401.14  198776.6 159974.35     2
## 5  175.9560468 172683.5  18767.30  191450.8 153916.19     3
## 6  160.3245966 165898.9  18343.06  184241.9 147555.81     4
## 7  146.0818013 159584.6  18086.49  177671.0 141498.06     4
## 8  133.1042967 153949.3  17908.27  171857.5 136040.99     4
## 9  121.2796778 149167.6  17846.35  167013.9 131321.23     4
## 10 110.5055255 145137.3  17876.06  163013.4 127261.25     4
## 11 100.6885192 141665.4  17998.89  159664.3 123666.50     5
## 12  91.7436287 138734.7  18206.27  156941.0 120528.44     5
## 13  83.5933775 136281.8  18452.06  154733.8 117829.73     5
## 14  76.1671723 134214.9  18705.62  152920.5 115509.26     5
## 15  69.4006906 132087.0  18978.33  151065.3 113108.68     6
## 16  63.2353245 129772.8  19269.87  149042.7 110502.94     6
## 17  57.6176726 127755.9  19537.97  147293.9 108217.95     6
## 18  52.4990774 126022.7  19739.18  145761.9 106283.50     6
## 19  47.8352040 124564.2  19916.25  144480.5 104647.98     6
## 20  43.5856563 123362.9  20094.92  143457.8 103267.96     6
## 21  39.7136268 122372.9  20271.04  142643.9 102101.83     6
## 22  36.1855776 121558.7  20440.22  141998.9 101118.44     6
## 23  32.9709506 120891.7  20601.33  141493.1 100290.39     6
## 24  30.0419022 120343.2  20751.92  141095.1  99591.28     6
## 25  27.3730624 119903.9  20889.34  140793.3  99014.58     6
## 26  24.9413150 119577.7  21010.37  140588.1  98567.31     6
## 27  22.7255973 119350.2  21116.23  140466.5  98234.01     6
## 28  20.7067179 119201.6  21214.78  140416.4  97986.83     6
## 29  18.8671902 119104.2  21309.50  140413.7  97794.71     6
## 30  17.1910810 119051.6  21398.12  140449.8  97653.53     7
## 31  15.6638727 119024.9  21479.18  140504.0  97545.68     7
## 32  14.2723374 119012.7  21554.03  140566.7  97458.68     7
## 33  13.0044223 119003.2  21624.35  140627.5  97378.85     9
## 34  11.8491453 118968.3  21684.96  140653.3  97283.38     9
## 35  10.7964999 118882.5  21731.24  140613.7  97151.26     9
## 36   9.8373686 118944.3  21758.06  140702.4  97186.24     9
## 37   8.9634439 119225.3  21777.49  141002.8  97447.83     9
## 38   8.1671562 119563.0  21802.69  141365.7  97760.36    11
## 39   7.4416086 119701.4  21852.82  141554.2  97848.54    11
## 40   6.7805166 119629.6  21946.50  141576.1  97683.05    12
## 41   6.1781542 119454.5  22069.52  141524.0  97384.99    12
## 42   5.6293040 119205.7  22209.74  141415.4  96995.94    13
## 43   5.1292121 118697.6  22197.69  140895.3  96499.88    13
## 44   4.6735471 118106.7  22051.55  140158.2  96055.13    13
## 45   4.2583620 117632.8  21917.24  139550.0  95715.56    13
## 46   3.8800609 117276.1  21814.83  139090.9  95461.26    13
## 47   3.5353670 116972.9  21728.84  138701.7  95244.06    13
## 48   3.2212947 116724.2  21660.79  138385.0  95063.38    13
## 49   2.9351238 116554.1  21605.05  138159.2  94949.08    13
## 50   2.6743755 116494.5  21557.43  138052.0  94937.12    13
## 51   2.4367913 116544.6  21498.87  138043.5  95045.76    13
## 52   2.2203135 116631.8  21418.60  138050.4  95213.20    14
## 53   2.0230670 116716.6  21303.68  138020.2  95412.88    15
## 54   1.8433433 116807.8  21195.72  138003.5  95612.06    15
## 55   1.6795857 116931.3  21130.47  138061.7  95800.80    17
## 56   1.5303760 117131.5  21120.89  138252.4  96010.59    17
## 57   1.3944216 117322.7  21162.41  138485.1  96160.32    17
## 58   1.2705450 117474.9  21226.91  138701.8  96248.02    17
## 59   1.1576733 117642.9  21293.12  138936.0  96349.74    17
## 60   1.0548288 117776.1  21361.40  139137.5  96414.70    17
## 61   0.9611207 117848.2  21423.38  139271.6  96424.79    17
## 62   0.8757374 117904.1  21476.97  139381.1  96427.16    17
## 63   0.7979393 117961.3  21526.65  139487.9  96434.61    17
## 64   0.7270526 118021.5  21572.56  139594.1  96448.95    17
## 65   0.6624632 118104.0  21611.59  139715.6  96492.43    18
## 66   0.6036118 118175.6  21649.29  139824.9  96526.29    18
## 67   0.5499886 118255.2  21683.25  139938.4  96571.90    18
## 68   0.5011291 118325.3  21715.84  140041.1  96609.44    17
## 69   0.4566102 118411.7  21744.05  140155.7  96667.62    18
## 70   0.4160462 118504.2  21772.39  140276.6  96731.85    18
## 71   0.3790858 118585.6  21795.77  140381.4  96789.81    18
## 72   0.3454089 118651.6  21818.81  140470.4  96832.75    18
## 73   0.3147237 118720.0  21839.34  140559.3  96880.67    18
## 74   0.2867645 118792.4  21854.77  140647.2  96937.61    18
## 75   0.2612891 118854.1  21863.33  140717.4  96990.76    18
## 76   0.2380769 118892.3  21874.29  140766.6  97018.04    18
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
