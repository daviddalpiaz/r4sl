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
## (Intercept) 172.720338908
## AtBat         0.099662970
## Hits          0.427613303
## HmRun         1.267838796
## Runs          0.676642660
## RBI           0.664847506
## Walks         0.887265880
## Years         2.665510665
## CAtBat        0.008472029
## CHits         0.033099124
## CHmRun        0.244686353
## CRuns         0.066354566
## CRBI          0.068696462
## CWalks        0.064445823
## LeagueN       4.803143606
## DivisionW   -27.147059583
## PutOuts       0.063770572
## Assists       0.008745578
## Errors       -0.209468235
## NewLeagueN    4.058198336
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
## (Intercept) 172.720338908
## AtBat         0.099662970
## Hits          0.427613303
## HmRun         1.267838796
## Runs          0.676642660
## RBI           0.664847506
## Walks         0.887265880
## Years         2.665510665
## CAtBat        0.008472029
## CHits         0.033099124
## CHmRun        0.244686353
## CRuns         0.066354566
## CRBI          0.068696462
## CWalks        0.064445823
## LeagueN       4.803143606
## DivisionW   -27.147059583
## PutOuts       0.063770572
## Assists       0.008745578
## Errors       -0.209468235
## NewLeagueN    4.058198336
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 787.2166
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 126796
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 452.0787 449.9788 449.4371 449.1687 448.8753 448.5547 448.2045
##  [8] 447.8222 447.4050 446.9501 446.4542 445.9140 445.3260 444.6866
## [15] 443.9916 443.2370 442.4186 441.5319 440.5725 439.5356 438.4166
## [22] 437.2108 435.9136 434.5207 433.0278 431.4312 429.7275 427.9140
## [29] 425.9885 423.9500 421.7984 419.5346 417.1609 414.6811 412.1006
## [36] 409.4262 406.6668 403.8320 400.9339 397.9858 395.0024 391.9995
## [43] 388.9936 386.0019 383.0417 380.1298 377.2826 374.5155 371.8424
## [50] 369.2754 366.8243 364.4981 362.3030 360.2428 358.3195 356.5331
## [57] 354.8817 353.3620 351.9694 350.6983 349.5421 348.4940 347.5467
## [64] 346.6928 345.9274 345.2424 344.6282 344.0795 343.5900 343.1554
## [71] 342.7710 342.4260 342.1173 341.8419 341.5951 341.3706 341.1642
## [78] 340.9724 340.7925 340.6200 340.4534 340.2877 340.1216 339.9541
## [85] 339.7803 339.6003 339.4135 339.2184 339.0127 338.7979 338.5741
## [92] 338.3407 338.0977 337.8471 337.5881 337.3240 337.0534 336.7802
## [99] 336.5037
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.5037
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 362.303
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
##  [1] 450.5679 440.7606 430.3006 421.3638 412.9710 404.4218 395.5359
##  [8] 387.8506 381.3558 375.8162 370.9353 366.7026 362.9737 359.4734
## [15] 356.1487 352.8230 349.9086 347.4602 345.4278 343.7404 342.3435
## [22] 341.1884 340.2327 339.4454 338.8000 338.2786 337.9030 337.6511
## [29] 337.5347 337.5083 337.5206 337.5713 337.6263 337.6591 337.6951
## [36] 337.9188 338.4341 338.9293 338.9852 338.6718 338.1644 337.5369
## [43] 336.6241 335.7354 335.0707 334.4993 333.9909 333.5872 333.2770
## [50] 333.1040 333.1571 333.4213 333.7891 334.1938 334.4690 334.6372
## [57] 334.8265 334.9333 335.0302 335.1413 335.2727 335.3885 335.5014
## [64] 335.6184 335.7337 335.8590 335.9870 336.0963 336.2418 336.3117
## [71] 336.4200 336.4876 336.5689 336.6723 336.6882 336.7382
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 333.104
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 356.1487
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
## 1  255.2820965 203011.5  24943.10  227954.6 178068.37     0
## 2  232.6035386 194269.9  24697.60  218967.5 169572.27     1
## 3  211.9396813 185158.6  23960.28  209118.9 161198.35     2
## 4  193.1115442 177547.4  23335.11  200882.5 154212.30     2
## 5  175.9560468 170545.1  22964.04  193509.1 147581.05     3
## 6  160.3245966 163557.0  22651.11  186208.1 140905.84     4
## 7  146.0818013 156448.6  22135.15  178583.8 134313.49     4
## 8  133.1042967 150428.1  21682.80  172110.9 128745.26     4
## 9  121.2796778 145432.2  21311.41  166743.6 124120.83     4
## 10 110.5055255 141237.8  20986.27  162224.1 120251.54     4
## 11 100.6885192 137593.0  20634.95  158228.0 116958.05     5
## 12  91.7436287 134470.8  20332.02  154802.8 114138.80     5
## 13  83.5933775 131749.9  20119.85  151869.7 111630.06     5
## 14  76.1671723 129221.1  19976.57  149197.7 109244.56     5
## 15  69.4006906 126841.9  19883.31  146725.2 106958.56     6
## 16  63.2353245 124484.1  19763.47  144247.6 104720.62     6
## 17  57.6176726 122436.1  19615.62  142051.7 102820.43     6
## 18  52.4990774 120728.6  19485.55  140214.2 101243.05     6
## 19  47.8352040 119320.4  19379.41  138699.8  99940.99     6
## 20  43.5856563 118157.5  19293.48  137451.0  98864.01     6
## 21  39.7136268 117199.1  19223.94  136423.0  97975.14     6
## 22  36.1855776 116409.5  19167.99  135577.5  97241.52     6
## 23  32.9709506 115758.3  19123.69  134882.0  96634.57     6
## 24  30.0419022 115223.2  19088.12  134311.3  96135.09     6
## 25  27.3730624 114785.4  19059.62  133845.1  95725.83     6
## 26  24.9413150 114432.4  19036.45  133468.9  95395.96     6
## 27  22.7255973 114178.4  19011.66  133190.1  95166.76     6
## 28  20.7067179 114008.3  18996.70  133005.0  95011.59     6
## 29  18.8671902 113929.7  19008.55  132938.2  94921.10     6
## 30  17.1910810 113911.9  19032.57  132944.4  94879.29     7
## 31  15.6638727 113920.2  19049.77  132969.9  94870.41     7
## 32  14.2723374 113954.4  19067.67  133022.1  94886.75     7
## 33  13.0044223 113991.5  19076.63  133068.1  94914.88     9
## 34  11.8491453 114013.7  19081.22  133094.9  94932.44     9
## 35  10.7964999 114038.0  19080.32  133118.3  94957.66     9
## 36   9.8373686 114189.1  19033.32  133222.5  95155.81     9
## 37   8.9634439 114537.6  18979.60  133517.3  95558.04     9
## 38   8.1671562 114873.1  18960.87  133833.9  95912.19    11
## 39   7.4416086 114911.0  18913.05  133824.0  95997.91    11
## 40   6.7805166 114698.6  18819.93  133518.5  95878.63    12
## 41   6.1781542 114355.2  18729.50  133084.7  95625.67    12
## 42   5.6293040 113931.2  18618.08  132549.3  95313.11    13
## 43   5.1292121 113315.8  18407.67  131723.5  94908.12    13
## 44   4.6735471 112718.3  18204.21  130922.5  94514.08    13
## 45   4.2583620 112272.4  18031.62  130304.0  94240.76    13
## 46   3.8800609 111889.8  17870.12  129759.9  94019.65    13
## 47   3.5353670 111549.9  17727.70  129277.6  93822.21    13
## 48   3.2212947 111280.4  17613.73  128894.2  93666.71    13
## 49   2.9351238 111073.6  17522.04  128595.6  93551.52    13
## 50   2.6743755 110958.3  17448.21  128406.5  93510.06    13
## 51   2.4367913 110993.7  17415.53  128409.2  93578.14    13
## 52   2.2203135 111169.7  17399.87  128569.6  93769.86    14
## 53   2.0230670 111415.1  17401.71  128816.9  94013.44    15
## 54   1.8433433 111685.5  17418.03  129103.5  94267.45    15
## 55   1.6795857 111869.5  17427.86  129297.4  94441.66    17
## 56   1.5303760 111982.1  17454.42  129436.5  94527.66    17
## 57   1.3944216 112108.8  17490.96  129599.7  94617.81    17
## 58   1.2705450 112180.3  17544.07  129724.4  94636.22    17
## 59   1.1576733 112245.2  17591.60  129836.8  94653.65    17
## 60   1.0548288 112319.7  17631.75  129951.4  94687.92    17
## 61   0.9611207 112407.8  17663.44  130071.2  94744.32    17
## 62   0.8757374 112485.5  17682.65  130168.1  94802.82    17
## 63   0.7979393 112561.2  17688.95  130250.1  94872.25    17
## 64   0.7270526 112639.7  17681.27  130321.0  94958.46    17
## 65   0.6624632 112717.1  17669.66  130386.8  95047.48    18
## 66   0.6036118 112801.3  17660.05  130461.3  95141.23    18
## 67   0.5499886 112887.3  17650.85  130538.1  95236.41    18
## 68   0.5011291 112960.7  17644.25  130605.0  95316.49    17
## 69   0.4566102 113058.5  17650.18  130708.7  95408.37    18
## 70   0.4160462 113105.5  17645.94  130751.5  95459.59    18
## 71   0.3790858 113178.4  17647.58  130826.0  95530.83    18
## 72   0.3454089 113223.9  17644.48  130868.4  95579.40    18
## 73   0.3147237 113278.6  17647.86  130926.5  95630.76    18
## 74   0.2867645 113348.2  17657.64  131005.9  95690.57    18
## 75   0.2612891 113358.9  17653.22  131012.1  95705.69    18
## 76   0.2380769 113392.6  17655.35  131047.9  95737.24    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   69.40069
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
