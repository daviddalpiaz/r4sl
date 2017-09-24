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
## (Intercept) 295.390327332
## AtBat         0.068388463
## Hits          0.267044986
## HmRun         0.934773737
## Runs          0.438045591
## RBI           0.446352643
## Walks         0.557852274
## Years         1.960352610
## CAtBat        0.005753196
## CHits         0.021745444
## CHmRun        0.162469784
## CRuns         0.043620173
## CRBI          0.045067462
## CWalks        0.045301836
## LeagueN       1.277898191
## DivisionW   -13.838679395
## PutOuts       0.034812580
## Assists       0.005233310
## Errors       -0.067928130
## NewLeagueN    1.404919256
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
## (Intercept) 295.390327332
## AtBat         0.068388463
## Hits          0.267044986
## HmRun         0.934773737
## Runs          0.438045591
## RBI           0.446352643
## Walks         0.557852274
## Years         1.960352610
## CAtBat        0.005753196
## CHits         0.021745444
## CHmRun        0.162469784
## CRuns         0.043620173
## CRBI          0.045067462
## CWalks        0.045301836
## LeagueN       1.277898191
## DivisionW   -13.838679395
## PutOuts       0.034812580
## Assists       0.005233310
## Errors       -0.067928130
## NewLeagueN    1.404919256
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 200.6497
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 145731.4
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.8615 450.4852 449.5741 449.3086 449.0183 448.7011 448.3548
##  [8] 447.9766 447.5641 447.1141 446.6238 446.0898 445.5086 444.8766
## [15] 444.1899 443.4445 442.6361 441.7606 440.8135 439.7902 438.6862
## [22] 437.4970 436.2183 434.8457 433.3754 431.8037 430.1275 428.3444
## [29] 426.4527 424.4514 422.3407 420.1222 417.7984 415.3735 412.8532
## [36] 410.2449 407.5576 404.8017 401.9894 399.1344 396.2518 393.3575
## [43] 390.4682 387.6013 384.7739 382.0030 379.3045 376.6937 374.1844
## [50] 371.7874 369.5125 367.3677 365.3584 363.4875 361.7563 360.1643
## [57] 358.7085 357.3852 356.1896 355.1146 354.1536 353.2995 352.5447
## [64] 351.8771 351.2984 350.7976 350.3684 350.0026 349.6865 349.4338
## [71] 349.2269 349.0590 348.9207 348.8227 348.7421 348.6920 348.6631
## [78] 348.6393 348.6363 348.6322 348.6377 348.6421 348.6474 348.6509
## [85] 348.6417 348.6324 348.6143 348.5798 348.5395 348.4873 348.4239
## [92] 348.3374 348.2642 348.1526 348.0617 347.9304 347.8297 347.6824
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 347.6824
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 387.6013
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
##  [1] 448.8334 440.4264 431.2809 423.0993 415.4927 407.9375 401.0250
##  [8] 394.4327 388.3587 383.2219 378.7939 374.8798 371.3376 368.3244
## [15] 365.6079 362.4892 359.3631 356.6786 354.4304 352.5458 350.9700
## [22] 349.6462 348.5403 347.6506 346.9539 346.4189 346.0175 345.7254
## [29] 345.5603 345.4590 345.3931 345.3429 345.3076 345.5024 345.8746
## [36] 346.4919 347.1588 347.6316 347.8408 347.9745 347.9488 347.7496
## [43] 347.1977 346.3634 345.5539 344.8785 344.3560 343.9750 343.7531
## [50] 343.6091 343.5475 343.6659 343.8335 344.0128 344.1873 344.3258
## [57] 344.4017 344.5266 344.5731 344.5244 344.4824 344.4816 344.5400
## [64] 344.6090 344.7407 344.8568 344.9853 345.1054 345.2767 345.4251
## [71] 345.7169 345.9395 346.1682 346.3588
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 343.5475
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 365.6079
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
## 1  255.2820965 201451.4  29781.03  231232.4 171670.4     0
## 2  232.6035386 193975.4  30091.88  224067.3 163883.5     1
## 3  211.9396813 186003.2  29435.80  215439.0 156567.4     2
## 4  193.1115442 179013.0  28530.02  207543.0 150483.0     2
## 5  175.9560468 172634.1  27777.20  200411.3 144857.0     3
## 6  160.3245966 166413.0  27120.80  193533.8 139292.2     4
## 7  146.0818013 160821.0  26594.63  187415.7 134226.4     4
## 8  133.1042967 155577.1  25801.95  181379.1 129775.2     4
## 9  121.2796778 150822.5  24859.77  175682.3 125962.7     4
## 10 110.5055255 146859.0  23968.53  170827.6 122890.5     4
## 11 100.6885192 143484.8  23184.18  166669.0 120300.6     5
## 12  91.7436287 140534.9  22545.64  163080.5 117989.3     5
## 13  83.5933775 137891.6  22037.29  159928.9 115854.3     5
## 14  76.1671723 135662.8  21601.90  157264.7 114060.9     5
## 15  69.4006906 133669.2  21225.85  154895.0 112443.3     6
## 16  63.2353245 131398.4  20692.36  152090.8 110706.1     6
## 17  57.6176726 129141.8  20096.02  149237.8 109045.8     6
## 18  52.4990774 127219.6  19582.90  146802.5 107636.7     6
## 19  47.8352040 125620.9  19156.64  144777.6 106464.3     6
## 20  43.5856563 124288.5  18799.86  143088.4 105488.7     6
## 21  39.7136268 123179.9  18503.95  141683.9 104676.0     6
## 22  36.1855776 122252.5  18254.16  140506.6 103998.3     6
## 23  32.9709506 121480.3  18045.54  139525.9 103434.8     6
## 24  30.0419022 120860.9  17874.49  138735.4 102986.4     6
## 25  27.3730624 120377.0  17735.62  138112.6 102641.4     6
## 26  24.9413150 120006.0  17617.12  137623.2 102388.9     6
## 27  22.7255973 119728.1  17524.44  137252.5 102203.6     6
## 28  20.7067179 119526.0  17458.60  136984.6 102067.4     6
## 29  18.8671902 119411.9  17402.30  136814.2 102009.7     6
## 30  17.1910810 119341.9  17352.07  136694.0 101989.8     7
## 31  15.6638727 119296.4  17311.24  136607.7 101985.2     7
## 32  14.2723374 119261.7  17277.49  136539.2 101984.2     7
## 33  13.0044223 119237.4  17247.43  136484.8 101989.9     9
## 34  11.8491453 119371.9  17262.40  136634.3 102109.5     9
## 35  10.7964999 119629.3  17322.25  136951.5 102307.0     9
## 36   9.8373686 120056.7  17420.31  137477.0 102636.3     9
## 37   8.9634439 120519.2  17521.59  138040.8 102997.6     9
## 38   8.1671562 120847.8  17564.94  138412.7 103282.8    11
## 39   7.4416086 120993.2  17633.00  138626.2 103360.2    11
## 40   6.7805166 121086.3  17712.54  138798.8 103373.7    12
## 41   6.1781542 121068.4  17744.82  138813.2 103323.5    12
## 42   5.6293040 120929.8  17746.17  138676.0 103183.6    13
## 43   5.1292121 120546.2  17617.96  138164.2 102928.3    13
## 44   4.6735471 119967.6  17420.48  137388.1 102547.1    13
## 45   4.2583620 119407.5  17221.84  136629.3 102185.6    13
## 46   3.8800609 118941.2  17057.26  135998.4 101883.9    13
## 47   3.5353670 118581.1  16912.75  135493.8 101668.3    13
## 48   3.2212947 118318.8  16783.44  135102.3 101535.4    13
## 49   2.9351238 118166.2  16663.47  134829.7 101502.7    13
## 50   2.6743755 118067.2  16551.01  134618.2 101516.2    13
## 51   2.4367913 118024.9  16428.49  134453.4 101596.4    13
## 52   2.2203135 118106.2  16305.26  134411.5 101801.0    14
## 53   2.0230670 118221.5  16188.79  134410.3 102032.7    15
## 54   1.8433433 118344.8  16095.88  134440.7 102249.0    15
## 55   1.6795857 118464.9  16014.86  134479.7 102450.0    17
## 56   1.5303760 118560.3  15951.74  134512.0 102608.5    17
## 57   1.3944216 118612.5  15907.88  134520.4 102704.7    17
## 58   1.2705450 118698.5  15858.74  134557.3 102839.8    17
## 59   1.1576733 118730.7  15794.89  134525.5 102935.8    17
## 60   1.0548288 118697.1  15729.19  134426.3 102967.9    17
## 61   0.9611207 118668.1  15671.03  134339.2 102997.1    17
## 62   0.8757374 118667.6  15617.17  134284.7 103050.4    17
## 63   0.7979393 118707.8  15563.02  134270.9 103144.8    17
## 64   0.7270526 118755.3  15515.41  134270.8 103239.9    17
## 65   0.6624632 118846.1  15467.93  134314.1 103378.2    18
## 66   0.6036118 118926.2  15430.21  134356.4 103496.0    18
## 67   0.5499886 119014.8  15395.94  134410.8 103618.9    18
## 68   0.5011291 119097.7  15363.57  134461.3 103734.2    17
## 69   0.4566102 119216.0  15343.40  134559.4 103872.6    18
## 70   0.4160462 119318.5  15328.35  134646.9 103990.2    18
## 71   0.3790858 119520.2  15346.15  134866.3 104174.0    18
## 72   0.3454089 119674.1  15354.23  135028.3 104319.9    18
## 73   0.3147237 119832.4  15370.94  135203.3 104461.4    18
## 74   0.2867645 119964.4  15383.05  135347.4 104581.3    18
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
