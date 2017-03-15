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
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
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
## (Intercept) 199.418112992
## AtBat         0.093426871
## Hits          0.389767264
## HmRun         1.212875008
## Runs          0.623229049
## RBI           0.618547530
## Walks         0.810467709
## Years         2.544170913
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295054
## DivisionW   -23.545192371
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203268
## NewLeagueN    3.313773178
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 588.9958
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 130404.9
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.2331 449.8250 448.8663 448.5740 448.2859 447.9710 447.6270
##  [8] 447.2516 446.8419 446.3951 445.9081 445.3778 444.8005 444.1727
## [15] 443.4905 442.7498 441.9466 441.0765 440.1351 439.1179 438.0202
## [22] 436.8376 435.5657 434.2001 432.7368 431.1722 429.5030 427.7266
## [29] 425.8412 423.8458 421.7401 419.5255 417.2042 414.7802 412.2586
## [36] 409.6464 406.9521 404.1857 401.3587 398.4843 395.5769 392.6517
## [43] 389.7249 386.8130 383.9328 381.1005 378.3316 375.6409 373.0415
## [50] 370.5444 368.1596 365.8945 363.7544 361.7431 359.8619 358.1102
## [57] 356.4859 354.9855 353.6041 352.3361 351.1751 350.1143 349.1469
## [64] 348.2647 347.4651 346.7403 346.0802 345.4806 344.9344 344.4404
## [71] 343.9950 343.5867 343.2118 342.8716 342.5598 342.2718 342.0005
## [78] 341.7493 341.5128 341.2863 341.0691 340.8573 340.6487 340.4415
## [85] 340.2352 340.0270 339.8150 339.5989 339.3784 339.1517 338.9191
## [92] 338.6799 338.4352 338.1849 337.9293 337.6696 337.4068 337.1409
## [99] 336.8747
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.8747
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 368.1596
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
## (Intercept) 2.220974e+02
## AtBat       .           
## Hits        1.129009e+00
## HmRun       .           
## Runs        .           
## RBI         .           
## Walks       1.172062e+00
## Years       .           
## CAtBat      .           
## CHits       .           
## CHmRun      .           
## CRuns       1.147170e-01
## CRBI        3.085475e-01
## CWalks      .           
## LeagueN     .           
## DivisionW   .           
## PutOuts     1.763115e-03
## Assists     .           
## Errors      .           
## NewLeagueN  .
```

```r
coef(fit_lasso_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  134.48030406
## AtBat         -1.67572220
## Hits           5.94122316
## HmRun          0.04746835
## Runs           .         
## RBI            .         
## Walks          4.95676182
## Years        -10.26657309
## CAtBat         .         
## CHits          .         
## CHmRun         0.56236426
## CRuns          0.70135135
## CRBI           0.38727139
## CWalks        -0.58111548
## LeagueN       32.92255640
## DivisionW   -119.37941356
## PutOuts        0.27580087
## Assists        0.19782326
## Errors        -2.26242857
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 180.1579
```

```r
coef(fit_lasso_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 2.220974e+02
## AtBat       .           
## Hits        1.129009e+00
## HmRun       .           
## Runs        .           
## RBI         .           
## Walks       1.172062e+00
## Years       .           
## CAtBat      .           
## CHits       .           
## CHmRun      .           
## CRuns       1.147170e-01
## CRBI        3.085475e-01
## CWalks      .           
## LeagueN     .           
## DivisionW   .           
## PutOuts     1.763115e-03
## Assists     .           
## Errors      .           
## NewLeagueN  .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 2.726099
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 130946.2
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 449.3907 440.9039 431.1582 423.0880 416.0486 408.4424 400.4321
##  [8] 393.3910 387.3527 382.3821 378.0877 374.0324 370.6170 367.7015
## [15] 364.6871 361.7823 359.0294 356.6992 354.7335 353.0715 351.6976
## [22] 350.5747 349.6509 348.9256 348.4845 348.2886 348.2189 348.2353
## [29] 348.3345 348.4834 348.6861 348.8928 349.1014 349.2990 349.6347
## [36] 350.3801 351.4217 352.2067 352.2355 351.9336 351.2357 350.5314
## [43] 349.8521 349.2288 348.5382 347.9193 347.4247 346.9946 346.6715
## [50] 346.3711 346.1898 346.1455 346.1753 346.2311 346.3099 346.5001
## [57] 346.7437 346.8380 346.9904 347.1218 347.2693 347.4506 347.6078
## [64] 347.7632 347.9102 348.0442 348.1709 348.2956 348.3987 348.5311
## [71] 348.5643 348.6822 348.7070 348.8064 348.8449 348.9190 348.9514
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 346.1455
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 378.0877
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
## 1  255.2820965 201952.0  26862.08  228814.1 175089.90     0
## 2  232.6035386 194396.2  27489.78  221886.0 166906.45     1
## 3  211.9396813 185897.4  26738.29  212635.7 159159.10     2
## 4  193.1115442 179003.5  25970.21  204973.7 153033.26     2
## 5  175.9560468 173096.4  25303.62  198400.0 147792.81     3
## 6  160.3245966 166825.2  24510.75  191336.0 142314.48     4
## 7  146.0818013 160345.8  23699.80  184045.7 136646.04     4
## 8  133.1042967 154756.5  23117.39  177873.9 131639.08     4
## 9  121.2796778 150042.1  22714.00  172756.1 127328.08     4
## 10 110.5055255 146216.1  22436.93  168653.0 123779.15     4
## 11 100.6885192 142950.3  22232.09  165182.4 120718.19     5
## 12  91.7436287 139900.2  22075.29  161975.5 117824.92     5
## 13  83.5933775 137357.0  22010.98  159367.9 115345.99     5
## 14  76.1671723 135204.4  22018.22  157222.6 113186.21     5
## 15  69.4006906 132996.7  22108.29  155105.0 110888.39     6
## 16  63.2353245 130886.4  22178.27  153064.7 108708.18     6
## 17  57.6176726 128902.1  22198.44  151100.6 106703.70     6
## 18  52.4990774 127234.3  22230.42  149464.8 105003.92     6
## 19  47.8352040 125835.9  22271.25  148107.1 103564.63     6
## 20  43.5856563 124659.5  22318.37  146977.9 102341.12     6
## 21  39.7136268 123691.2  22377.90  146069.1 101313.33     6
## 22  36.1855776 122902.7  22446.04  145348.7 100456.61     6
## 23  32.9709506 122255.7  22519.27  144775.0  99736.46     6
## 24  30.0419022 121749.0  22600.66  144349.7  99148.38     6
## 25  27.3730624 121441.4  22708.08  144149.5  98733.35     6
## 26  24.9413150 121304.9  22826.04  144131.0  98478.88     6
## 27  22.7255973 121256.4  22929.90  144186.3  98326.51     6
## 28  20.7067179 121267.8  23018.68  144286.5  98249.16     6
## 29  18.8671902 121336.9  23093.90  144430.8  98243.03     6
## 30  17.1910810 121440.7  23161.20  144601.9  98279.50     7
## 31  15.6638727 121582.0  23222.19  144804.2  98359.81     7
## 32  14.2723374 121726.2  23282.64  145008.8  98443.58     7
## 33  13.0044223 121871.8  23340.43  145212.2  98531.33     9
## 34  11.8491453 122009.8  23396.08  145405.9  98613.71     9
## 35  10.7964999 122244.4  23493.00  145737.4  98751.42     9
## 36   9.8373686 122766.2  23686.75  146453.0  99079.48     9
## 37   8.9634439 123497.2  23915.54  147412.8  99581.70     9
## 38   8.1671562 124049.5  24084.20  148133.7  99965.33    11
## 39   7.4416086 124069.8  24288.80  148358.6  99781.02    11
## 40   6.7805166 123857.3  24463.44  148320.7  99393.83    12
## 41   6.1781542 123366.5  24546.81  147913.3  98819.73    12
## 42   5.6293040 122872.3  24594.97  147467.3  98277.33    13
## 43   5.1292121 122396.5  24613.26  147009.7  97783.23    13
## 44   4.6735471 121960.8  24632.83  146593.6  97327.95    13
## 45   4.2583620 121478.9  24682.13  146161.0  96796.76    13
## 46   3.8800609 121047.8  24736.80  145784.6  96311.02    13
## 47   3.5353670 120703.9  24788.86  145492.8  95915.05    13
## 48   3.2212947 120405.3  24833.15  145238.4  95572.12    13
## 49   2.9351238 120181.1  24884.46  145065.6  95296.67    13
## 50   2.6743755 119972.9  24950.15  144923.1  95022.78    13
## 51   2.4367913 119847.3  25022.28  144869.6  94825.07    13
## 52   2.2203135 119816.7  25072.85  144889.6  94743.89    14
## 53   2.0230670 119837.3  25091.98  144929.3  94745.36    15
## 54   1.8433433 119876.0  25099.61  144975.6  94776.40    15
## 55   1.6795857 119930.5  25116.07  145046.6  94814.47    17
## 56   1.5303760 120062.3  25137.98  145200.3  94924.34    17
## 57   1.3944216 120231.2  25157.28  145388.5  95073.92    17
## 58   1.2705450 120296.6  25143.48  145440.1  95153.13    17
## 59   1.1576733 120402.3  25137.09  145539.4  95265.22    17
## 60   1.0548288 120493.5  25133.27  145626.8  95360.25    17
## 61   0.9611207 120596.0  25131.40  145727.4  95464.60    17
## 62   0.8757374 120721.9  25135.23  145857.1  95586.67    17
## 63   0.7979393 120831.2  25135.63  145966.8  95695.55    17
## 64   0.7270526 120939.2  25134.50  146073.7  95804.72    17
## 65   0.6624632 121041.5  25134.34  146175.8  95907.16    18
## 66   0.6036118 121134.8  25135.44  146270.2  95999.32    18
## 67   0.5499886 121223.0  25138.73  146361.7  96084.27    18
## 68   0.5011291 121309.8  25141.27  146451.1  96168.56    17
## 69   0.4566102 121381.7  25142.07  146523.7  96239.58    18
## 70   0.4160462 121473.9  25141.17  146615.1  96332.72    18
## 71   0.3790858 121497.0  25142.10  146639.1  96354.95    18
## 72   0.3454089 121579.3  25139.37  146718.6  96439.91    18
## 73   0.3147237 121596.6  25138.07  146734.6  96458.51    18
## 74   0.2867645 121665.9  25138.48  146804.4  96527.41    18
## 75   0.2612891 121692.8  25136.64  146829.4  96556.16    18
## 76   0.2380769 121744.5  25138.78  146883.3  96605.69    18
## 77   0.2169268 121767.1  25139.94  146907.0  96627.17    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.220313   100.6885
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
