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
##  [1] 450.4982 448.7733 448.0647 447.7956 447.5013 447.1798 446.8286
##  [8] 446.4452 446.0269 445.5707 445.0734 444.5318 443.9423 443.3011
## [15] 442.6044 441.8479 441.0276 440.1388 439.1772 438.1381 437.0168
## [22] 435.8086 434.5091 433.1138 431.6187 430.0199 428.3142 426.4989
## [29] 424.5721 422.5327 420.3807 418.1173 415.7450 413.2676 410.6907
## [36] 408.0215 405.2689 402.4430 399.5560 396.6216 393.6547 390.6712
## [43] 387.6881 384.7226 381.7921 378.9137 376.1037 373.3776 370.7492
## [50] 368.2306 365.8308 363.5589 361.4208 359.4197 357.5573 355.8329
## [57] 354.2444 352.7880 351.4585 350.2500 349.1556 348.1681 347.2802
## [64] 346.4831 345.7742 345.1422 344.5821 344.0850 343.6449 343.2591
## [71] 342.9220 342.6275 342.3666 342.1387 341.9407 341.7675 341.6142
## [78] 341.4790 341.3559 341.2420 341.1401 341.0407 340.9445 340.8506
## [85] 340.7546 340.6569 340.5566 340.4509 340.3412 340.2264 340.1054
## [92] 339.9779 339.8461 339.7097 339.5689 339.4254 339.2793 339.1331
## [99] 338.9868
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.9868
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.7492
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
##  [1] 449.3055 441.6712 431.4396 422.9493 415.3381 407.5896 399.5736
##  [8] 392.1650 385.8404 380.5873 376.0771 372.1098 368.5042 365.4019
## [15] 362.2956 359.1721 356.3461 353.8683 351.7531 350.0077 348.5740
## [22] 347.4013 346.4278 345.6222 344.9814 344.5355 344.2753 344.0853
## [29] 344.0121 344.0111 344.0355 344.0615 344.0967 344.1666 344.4989
## [36] 345.1990 346.0578 346.8125 346.7021 346.0397 345.3283 344.7106
## [43] 344.0665 343.4820 342.7107 342.0981 341.6225 341.2325 340.9545
## [50] 340.8271 340.8136 341.0235 341.3786 341.8750 342.3496 342.7390
## [57] 343.1381 343.5468 343.9330 344.1475 344.4003 344.6306 344.8303
## [64] 345.0278 345.2465 345.5343 345.7249 345.9726 346.1414 346.3553
## [71] 346.5089 346.6715 346.8391 346.9434
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 340.8136
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 368.5042
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
## 1  255.2820965 201875.4  27253.04  229128.5 174622.41     0
## 2  232.6035386 195073.5  27750.55  222824.0 167322.94     1
## 3  211.9396813 186140.1  26335.61  212475.7 159804.48     2
## 4  193.1115442 178886.1  25082.88  203969.0 153803.19     2
## 5  175.9560468 172505.8  24138.58  196644.4 148367.19     3
## 6  160.3245966 166129.3  23399.98  189529.2 142729.28     4
## 7  146.0818013 159659.0  22547.11  182206.1 137111.92     4
## 8  133.1042967 153793.4  21930.74  175724.1 131862.64     4
## 9  121.2796778 148872.8  21504.64  170377.5 127368.19     4
## 10 110.5055255 144846.7  21170.61  166017.3 123676.09     4
## 11 100.6885192 141434.0  20907.06  162341.0 120526.92     5
## 12  91.7436287 138465.7  20777.71  159243.4 117688.02     5
## 13  83.5933775 135795.3  20722.07  156517.4 115073.24     5
## 14  76.1671723 133518.6  20712.16  154230.7 112806.39     5
## 15  69.4006906 131258.1  20700.91  151959.1 110557.22     6
## 16  63.2353245 129004.6  20667.82  149672.4 108336.81     6
## 17  57.6176726 126982.6  20603.65  147586.2 106378.91     6
## 18  52.4990774 125222.8  20507.99  145730.8 104714.81     6
## 19  47.8352040 123730.3  20421.85  144152.1 103308.42     6
## 20  43.5856563 122505.4  20365.21  142870.6 102140.16     6
## 21  39.7136268 121503.9  20329.96  141833.8 101173.90     6
## 22  36.1855776 120687.7  20312.25  140999.9 100375.41     6
## 23  32.9709506 120012.2  20308.48  140320.7  99703.77     6
## 24  30.0419022 119454.7  20314.81  139769.5  99139.90     6
## 25  27.3730624 119012.2  20325.94  139338.1  98686.25     6
## 26  24.9413150 118704.7  20336.39  139041.1  98368.30     6
## 27  22.7255973 118525.5  20343.40  138868.9  98182.07     6
## 28  20.7067179 118394.7  20354.65  138749.3  98040.03     6
## 29  18.8671902 118344.3  20363.34  138707.6  97980.95     6
## 30  17.1910810 118343.7  20376.06  138719.7  97967.60     7
## 31  15.6638727 118360.4  20393.75  138754.2  97966.67     7
## 32  14.2723374 118378.3  20413.22  138791.5  97965.06     7
## 33  13.0044223 118402.5  20434.32  138836.9  97968.22     9
## 34  11.8491453 118450.7  20455.57  138906.2  97995.10     9
## 35  10.7964999 118679.5  20499.59  139179.1  98179.90     9
## 36   9.8373686 119162.4  20599.36  139761.7  98563.00     9
## 37   8.9634439 119756.0  20705.72  140461.7  99050.27     9
## 38   8.1671562 120278.9  20763.38  141042.3  99515.52    11
## 39   7.4416086 120202.3  20791.64  140994.0  99410.71    11
## 40   6.7805166 119743.5  20785.55  140529.0  98957.92    12
## 41   6.1781542 119251.6  20813.92  140065.5  98437.68    12
## 42   5.6293040 118825.4  20849.41  139674.8  97975.97    13
## 43   5.1292121 118381.8  20898.94  139280.7  97482.82    13
## 44   4.6735471 117979.9  20953.68  138933.6  97026.20    13
## 45   4.2583620 117450.6  20939.62  138390.2  96511.01    13
## 46   3.8800609 117031.1  20910.33  137941.5  96120.81    13
## 47   3.5353670 116705.9  20884.92  137590.8  95821.01    13
## 48   3.2212947 116439.6  20860.95  137300.6  95578.65    13
## 49   2.9351238 116250.0  20835.54  137085.5  95414.46    13
## 50   2.6743755 116163.1  20806.20  136969.3  95356.93    13
## 51   2.4367913 116153.9  20772.62  136926.5  95381.31    13
## 52   2.2203135 116297.0  20719.52  137016.5  95577.48    14
## 53   2.0230670 116539.3  20638.55  137177.9  95900.80    15
## 54   1.8433433 116878.5  20563.43  137442.0  96315.11    15
## 55   1.6795857 117203.3  20502.48  137705.7  96700.78    17
## 56   1.5303760 117470.0  20438.16  137908.2  97031.86    17
## 57   1.3944216 117743.8  20362.84  138106.6  97380.92    17
## 58   1.2705450 118024.4  20314.55  138338.9  97709.82    17
## 59   1.1576733 118289.9  20279.95  138569.8  98009.93    17
## 60   1.0548288 118437.5  20234.44  138671.9  98203.05    17
## 61   0.9611207 118611.6  20196.50  138808.1  98415.10    17
## 62   0.8757374 118770.3  20164.94  138935.2  98605.31    17
## 63   0.7979393 118907.9  20133.46  139041.4  98774.44    17
## 64   0.7270526 119044.2  20109.48  139153.7  98934.71    17
## 65   0.6624632 119195.2  20088.59  139283.8  99106.59    18
## 66   0.6036118 119394.0  20070.94  139464.9  99323.04    18
## 67   0.5499886 119525.7  20056.00  139581.7  99469.71    18
## 68   0.5011291 119697.1  20041.33  139738.4  99655.75    17
## 69   0.4566102 119813.9  20031.01  139844.9  99782.86    18
## 70   0.4160462 119962.0  20030.46  139992.5  99931.56    18
## 71   0.3790858 120068.4  20030.10  140098.5 100038.29    18
## 72   0.3454089 120181.1  20030.07  140211.2 100151.03    18
## 73   0.3147237 120297.4  20026.33  140323.7 100271.03    18
## 74   0.2867645 120369.7  20035.64  140405.3 100334.05    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   83.59338
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
