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
##  [1] 452.5145 450.7514 450.0094 449.7412 449.4480 449.1277 448.7778
##  [8] 448.3957 447.9789 447.5242 447.0287 446.4890 445.9014 445.2625
## [15] 444.5681 443.8141 442.9963 442.1104 441.1518 440.1158 438.9978
## [22] 437.7931 436.4971 435.1056 433.6142 432.0192 430.3173 428.5058
## [29] 426.5826 424.5466 422.3976 420.1368 417.7665 415.2904 412.7139
## [36] 410.0439 407.2893 404.4597 401.5673 398.6253 395.6485 392.6526
## [43] 389.6542 386.6702 383.7180 380.8143 377.9754 375.2165 372.5514
## [50] 369.9920 367.5481 365.2276 363.0374 360.9808 359.0595 357.2733
## [57] 355.6201 354.0965 352.6977 351.4181 350.2511 349.1898 348.2272
## [64] 347.3551 346.5676 345.8629 345.2282 344.6565 344.1432 343.6824
## [71] 343.2743 342.9079 342.5784 342.2835 342.0190 341.7782 341.5596
## [78] 341.3581 341.1728 340.9963 340.8298 340.6664 340.5074 340.3465
## [85] 340.1847 340.0196 339.8487 339.6709 339.4854 339.2912 339.0878
## [92] 338.8760 338.6552 338.4256 338.1880 337.9430 337.6921 337.4372
## [99] 337.1779
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.1779
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 365.2276
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
##  [1] 449.7439 440.3343 430.1494 421.8417 414.0386 405.0786 396.7412
##  [8] 389.4224 383.1916 378.0604 373.8698 370.1640 366.6519 363.4254
## [15] 360.4878 357.7337 355.2129 352.8792 350.9937 349.4809 348.2668
## [22] 347.3048 346.5250 345.8978 345.4206 345.1100 344.9257 344.8218
## [29] 344.7772 344.8127 344.8987 344.9933 345.0522 345.2725 345.8357
## [36] 346.5968 347.3777 347.9966 348.3327 348.0536 347.4527 346.8410
## [43] 346.2787 345.5574 344.9576 344.3945 343.8466 343.4467 343.1648
## [50] 342.9698 342.9220 343.0418 343.1705 343.4181 343.7112 343.9070
## [57] 344.1016 344.3166 344.5227 344.7140 344.8692 345.0162 345.1541
## [64] 345.2525 345.3812 345.5086 345.6265 345.7375 345.8483 345.9457
## [71] 346.0357
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.922
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 370.164
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
## 1  255.2820965 202269.6  27793.06  230062.6 174476.50     0
## 2  232.6035386 193894.3  27699.58  221593.9 166194.75     1
## 3  211.9396813 185028.5  26915.98  211944.5 158112.52     2
## 4  193.1115442 177950.4  26298.20  204248.6 151652.19     2
## 5  175.9560468 171427.9  25710.56  197138.5 145717.39     3
## 6  160.3245966 164088.7  25450.56  189539.3 138638.15     4
## 7  146.0818013 157403.6  25333.03  182736.6 132070.59     4
## 8  133.1042967 151649.8  25178.39  176828.2 126471.41     4
## 9  121.2796778 146835.8  25007.92  171843.8 121827.91     4
## 10 110.5055255 142929.6  24881.46  167811.1 118048.17     4
## 11 100.6885192 139778.7  24805.46  164584.1 114973.19     5
## 12  91.7436287 137021.4  24795.44  161816.9 112225.98     5
## 13  83.5933775 134433.6  24811.68  159245.3 109621.91     5
## 14  76.1671723 132078.0  24665.93  156743.9 107412.08     5
## 15  69.4006906 129951.5  24572.70  154524.2 105378.79     6
## 16  63.2353245 127973.4  24525.22  152498.6 103448.17     6
## 17  57.6176726 126176.2  24448.11  150624.3 101728.09     6
## 18  52.4990774 124523.7  24259.22  148783.0 100264.51     6
## 19  47.8352040 123196.6  24087.65  147284.2  99108.94     6
## 20  43.5856563 122136.9  23936.25  146073.2  98200.65     6
## 21  39.7136268 121289.7  23803.58  145093.3  97486.16     6
## 22  36.1855776 120620.6  23685.50  144306.1  96935.14     6
## 23  32.9709506 120079.6  23585.84  143665.4  96493.73     6
## 24  30.0419022 119645.3  23501.66  143147.0  96143.65     6
## 25  27.3730624 119315.4  23430.46  142745.8  95884.91     6
## 26  24.9413150 119100.9  23365.38  142466.3  95735.57     6
## 27  22.7255973 118973.8  23298.03  142271.8  95675.72     6
## 28  20.7067179 118902.1  23232.66  142134.7  95669.40     6
## 29  18.8671902 118871.3  23174.12  142045.4  95697.20     6
## 30  17.1910810 118895.8  23122.12  142017.9  95773.66     7
## 31  15.6638727 118955.1  23082.16  142037.3  95872.95     7
## 32  14.2723374 119020.4  23044.15  142064.5  95976.24     7
## 33  13.0044223 119061.0  23012.71  142073.7  96048.28     9
## 34  11.8491453 119213.1  22990.88  142203.9  96222.19     9
## 35  10.7964999 119602.4  22978.36  142580.7  96624.01     9
## 36   9.8373686 120129.4  22976.52  143105.9  97152.84     9
## 37   8.9634439 120671.2  22954.81  143626.1  97716.44     9
## 38   8.1671562 121101.6  22919.78  144021.4  98181.87    11
## 39   7.4416086 121335.7  22897.55  144233.2  98438.13    11
## 40   6.7805166 121141.3  22962.56  144103.8  98178.72    12
## 41   6.1781542 120723.4  22881.90  143605.3  97841.51    12
## 42   5.6293040 120298.7  22736.75  143035.4  97561.91    13
## 43   5.1292121 119908.9  22543.46  142452.4  97365.47    13
## 44   4.6735471 119409.9  22343.79  141753.7  97066.14    13
## 45   4.2583620 118995.7  22179.13  141174.9  96816.60    13
## 46   3.8800609 118607.6  22026.48  140634.1  96581.12    13
## 47   3.5353670 118230.5  21843.91  140074.4  96386.58    13
## 48   3.2212947 117955.7  21680.43  139636.1  96275.24    13
## 49   2.9351238 117762.1  21528.89  139291.0  96233.22    13
## 50   2.6743755 117628.3  21396.32  139024.6  96231.93    13
## 51   2.4367913 117595.5  21270.85  138866.3  96324.64    13
## 52   2.2203135 117677.7  21161.94  138839.6  96515.73    14
## 53   2.0230670 117766.0  21028.28  138794.2  96737.68    15
## 54   1.8433433 117936.0  20918.83  138854.8  97017.17    15
## 55   1.6795857 118137.4  20825.72  138963.1  97311.70    17
## 56   1.5303760 118272.0  20688.87  138960.9  97583.16    17
## 57   1.3944216 118405.9  20582.31  138988.2  97823.57    17
## 58   1.2705450 118553.9  20493.46  139047.4  98060.44    17
## 59   1.1576733 118695.9  20423.29  139119.2  98272.59    17
## 60   1.0548288 118827.7  20367.99  139195.7  98459.72    17
## 61   0.9611207 118934.7  20329.27  139264.0  98605.47    17
## 62   0.8757374 119036.2  20293.17  139329.3  98742.98    17
## 63   0.7979393 119131.3  20270.96  139402.3  98860.36    17
## 64   0.7270526 119199.3  20254.66  139453.9  98944.61    17
## 65   0.6624632 119288.2  20249.54  139537.7  99038.64    18
## 66   0.6036118 119376.2  20241.16  139617.3  99135.02    18
## 67   0.5499886 119457.6  20239.17  139696.8  99218.48    18
## 68   0.5011291 119534.4  20229.36  139763.8  99305.08    17
## 69   0.4566102 119611.1  20240.83  139851.9  99370.24    18
## 70   0.4160462 119678.5  20226.08  139904.5  99452.37    18
## 71   0.3790858 119740.7  20241.17  139981.9  99499.56    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   91.74363
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
