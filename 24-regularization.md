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
## (Intercept) 240.682852262
## AtBat         0.083042199
## Hits          0.334990595
## HmRun         1.105720594
## Runs          0.542496738
## RBI           0.545363022
## Walks         0.698162048
## Years         2.316374820
## CAtBat        0.006988341
## CHits         0.026721778
## CHmRun        0.198876945
## CRuns         0.053594554
## CRBI          0.055409116
## CWalks        0.054405147
## LeagueN       2.463634059
## DivisionW   -18.860043802
## PutOuts       0.046088692
## Assists       0.006674057
## Errors       -0.112913895
## NewLeagueN    2.358350957
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
## (Intercept) 240.682852262
## AtBat         0.083042199
## Hits          0.334990595
## HmRun         1.105720594
## Runs          0.542496738
## RBI           0.545363022
## Walks         0.698162048
## Years         2.316374820
## CAtBat        0.006988341
## CHits         0.026721778
## CHmRun        0.198876945
## CRuns         0.053594554
## CRBI          0.055409116
## CWalks        0.054405147
## LeagueN       2.463634059
## DivisionW   -18.860043802
## PutOuts       0.046088692
## Assists       0.006674057
## Errors       -0.112913895
## NewLeagueN    2.358350957
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 375.1832
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 136525.7
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.4063 450.0985 449.2420 448.8522 448.5617 448.2444 447.8977
##  [8] 447.5193 447.1064 446.6562 446.1654 445.6309 445.0491 444.4165
## [15] 443.7290 442.9826 442.1732 441.2965 440.3479 439.3229 438.2170
## [22] 437.0255 435.7441 434.3684 432.8944 431.3185 429.6374 427.8486
## [29] 425.9503 423.9413 421.8219 419.5931 417.2576 414.8193 412.2837
## [36] 409.6581 406.9510 404.1728 401.3354 398.4524 395.5384 392.6094
## [43] 389.6819 386.7729 383.8997 381.0789 378.3269 375.6585 373.0877
## [50] 370.6257 368.2822 366.0661 363.9821 362.0345 360.2247 358.5523
## [57] 357.0150 355.6093 354.3304 353.1725 352.1290 351.1932 350.3578
## [64] 349.6137 348.9587 348.3873 347.8856 347.4503 347.0735 346.7542
## [71] 346.4881 346.2637 346.0790 345.9303 345.8137 345.7244 345.6553
## [78] 345.6107 345.5784 345.5624 345.5522 345.5529 345.5574 345.5630
## [85] 345.5695 345.5769 345.5790 345.5780 345.5710 345.5595 345.5397
## [92] 345.5167 345.4829 345.4464 345.3983 345.3502 345.2941 345.2350
## [99] 345.1712
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 345.1712
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 375.6585
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
##  [1] 451.4045 441.9423 431.6646 423.0623 414.9438 406.5346 398.2924
##  [8] 390.7582 384.2628 378.8188 374.3530 370.4526 366.8722 363.7002
## [15] 360.8262 358.1919 355.6313 353.3416 351.3434 349.6869 348.3152
## [22] 347.1794 346.2339 345.4462 344.7981 344.3002 344.0413 343.8898
## [29] 343.8127 343.8170 343.8784 343.9523 344.0058 344.0154 344.0312
## [36] 344.3641 344.9328 345.4778 345.6396 345.3448 344.6628 343.8008
## [43] 342.9271 342.1193 341.4617 340.9166 340.5281 340.2370 340.0535
## [50] 339.9534 339.8859 339.9589 340.0353 340.1239 340.3222 340.5767
## [57] 340.7474 340.8535 340.9203 341.0124 341.1598 341.3557 341.6022
## [64] 341.8722 342.1375 342.3706 342.6429 342.8454 343.0983 343.2965
## [71] 343.5266 343.7036 343.8587
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 339.8859
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 374.353
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
## 1  255.2820965 203766.0  34695.84  238461.9 169070.16     0
## 2  232.6035386 195313.0  34437.74  229750.7 160875.23     1
## 3  211.9396813 186334.3  33533.40  219867.7 152800.94     2
## 4  193.1115442 178981.7  32796.91  211778.6 146184.84     2
## 5  175.9560468 172178.4  32239.05  204417.4 139939.32     3
## 6  160.3245966 165270.4  31753.83  197024.2 133516.54     4
## 7  146.0818013 158636.8  31303.30  189940.1 127333.54     4
## 8  133.1042967 152692.0  30927.47  183619.4 121764.48     4
## 9  121.2796778 147657.9  30639.56  178297.5 117018.37     4
## 10 110.5055255 143503.7  30392.65  173896.3 113111.05     4
## 11 100.6885192 140140.2  30213.82  170354.0 109926.34     5
## 12  91.7436287 137235.1  30062.96  167298.1 107172.15     5
## 13  83.5933775 134595.2  29932.59  164527.8 104662.63     5
## 14  76.1671723 132277.8  29799.55  162077.4 102478.28     5
## 15  69.4006906 130195.5  29635.90  159831.4 100559.63     6
## 16  63.2353245 128301.4  29482.79  157784.2  98818.63     6
## 17  57.6176726 126473.6  29289.99  155763.6  97183.63     6
## 18  52.4990774 124850.3  29052.55  153902.9  95797.75     6
## 19  47.8352040 123442.2  28782.67  152224.9  94659.54     6
## 20  43.5856563 122280.9  28542.38  150823.3  93738.53     6
## 21  39.7136268 121323.5  28328.38  149651.9  92995.10     6
## 22  36.1855776 120533.6  28136.24  148669.8  92397.32     6
## 23  32.9709506 119877.9  27964.64  147842.5  91913.26     6
## 24  30.0419022 119333.1  27810.17  147143.3  91522.91     6
## 25  27.3730624 118885.7  27671.00  146556.7  91214.70     6
## 26  24.9413150 118542.6  27540.91  146083.5  91001.71     6
## 27  22.7255973 118364.4  27404.93  145769.4  90959.51     6
## 28  20.7067179 118260.2  27278.60  145538.8  90981.60     6
## 29  18.8671902 118207.2  27158.35  145365.5  91048.85     6
## 30  17.1910810 118210.1  27046.59  145256.7  91163.53     7
## 31  15.6638727 118252.3  26950.34  145202.7  91301.99     7
## 32  14.2723374 118303.2  26863.03  145166.2  91440.15     7
## 33  13.0044223 118340.0  26787.90  145127.9  91552.12     9
## 34  11.8491453 118346.6  26717.56  145064.1  91629.02     9
## 35  10.7964999 118357.5  26648.75  145006.2  91708.71     9
## 36   9.8373686 118586.6  26551.90  145138.6  92034.74     9
## 37   8.9634439 118978.6  26442.36  145421.0  92536.28     9
## 38   8.1671562 119354.9  26353.46  145708.4  93001.45    11
## 39   7.4416086 119466.7  26289.80  145756.5  93176.92    11
## 40   6.7805166 119263.1  26215.81  145478.9  93047.25    12
## 41   6.1781542 118792.4  26097.72  144890.2  92694.70    12
## 42   5.6293040 118199.0  25860.28  144059.3  92338.71    13
## 43   5.1292121 117599.0  25650.42  143249.4  91948.54    13
## 44   4.6735471 117045.6  25483.06  142528.7  91562.57    13
## 45   4.2583620 116596.1  25323.71  141919.8  91272.39    13
## 46   3.8800609 116224.1  25165.31  141389.4  91058.82    13
## 47   3.5353670 115959.4  25024.60  140984.0  90934.78    13
## 48   3.2212947 115761.2  24906.21  140667.5  90855.03    13
## 49   2.9351238 115636.4  24811.55  140448.0  90824.86    13
## 50   2.6743755 115568.3  24730.85  140299.2  90837.48    13
## 51   2.4367913 115522.5  24661.39  140183.8  90861.06    13
## 52   2.2203135 115572.0  24594.33  140166.4  90977.71    14
## 53   2.0230670 115624.0  24550.04  140174.0  91073.94    15
## 54   1.8433433 115684.2  24534.20  140218.4  91150.04    15
## 55   1.6795857 115819.2  24534.60  140353.8  91284.57    17
## 56   1.5303760 115992.5  24508.17  140500.6  91484.29    17
## 57   1.3944216 116108.8  24451.64  140560.4  91657.14    17
## 58   1.2705450 116181.1  24401.79  140582.9  91779.32    17
## 59   1.1576733 116226.7  24358.98  140585.6  91867.68    17
## 60   1.0548288 116289.4  24326.03  140615.5  91963.39    17
## 61   0.9611207 116390.0  24300.46  140690.5  92089.57    17
## 62   0.8757374 116523.7  24275.43  140799.1  92248.27    17
## 63   0.7979393 116692.1  24257.85  140949.9  92434.21    17
## 64   0.7270526 116876.6  24254.95  141131.5  92621.63    17
## 65   0.6624632 117058.1  24253.07  141311.2  92805.02    18
## 66   0.6036118 117217.6  24251.90  141469.5  92965.70    18
## 67   0.5499886 117404.2  24248.74  141652.9  93155.43    18
## 68   0.5011291 117543.0  24251.32  141794.3  93291.68    17
## 69   0.4566102 117716.4  24252.40  141968.8  93464.04    18
## 70   0.4160462 117852.5  24252.23  142104.7  93600.23    18
## 71   0.3790858 118010.5  24252.12  142262.7  93758.42    18
## 72   0.3454089 118132.2  24250.54  142382.7  93881.62    18
## 73   0.3147237 118238.8  24250.67  142489.5  93988.16    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   100.6885
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
