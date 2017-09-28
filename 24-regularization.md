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
## (Intercept) 147.229939657
## AtBat         0.105066772
## Hits          0.466242141
## HmRun         1.305566983
## Runs          0.728770399
## RBI           0.707973644
## Walks         0.964741609
## Years         2.755005487
## CAtBat        0.009010094
## CHits         0.035609341
## CHmRun        0.262373080
## CRuns         0.071371506
## CRBI          0.073956620
## CWalks        0.067634293
## LeagueN       6.031739802
## DivisionW   -31.146275021
## PutOuts       0.072008377
## Assists       0.009677269
## Errors       -0.265353795
## NewLeagueN    4.871395889
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
## (Intercept) 147.229939657
## AtBat         0.105066772
## Hits          0.466242141
## HmRun         1.305566983
## Runs          0.728770399
## RBI           0.707973644
## Walks         0.964741609
## Years         2.755005487
## CAtBat        0.009010094
## CHits         0.035609341
## CHmRun        0.262373080
## CRuns         0.071371506
## CRBI          0.073956620
## CWalks        0.067634293
## LeagueN       6.031739802
## DivisionW   -31.146275021
## PutOuts       0.072008377
## Assists       0.009677269
## Errors       -0.265353795
## NewLeagueN    4.871395889
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 1041.85
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 123586
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.5890 449.8181 449.2231 448.8530 448.5594 448.2385 447.8880
##  [8] 447.5054 447.0878 446.6324 446.1360 445.5953 445.0066 444.3664
## [15] 443.6707 442.9151 442.0956 441.2078 440.2469 439.2085 438.0876
## [22] 436.8798 435.5803 434.1847 432.6888 431.0887 429.3811 427.5630
## [29] 425.6325 423.5883 421.4301 419.1589 416.7770 414.2879 411.6970
## [36] 409.0110 406.2385 403.3893 400.4752 397.5095 394.5067 391.4825
## [43] 388.4533 385.4361 382.4481 379.5061 376.6262 373.8237 371.1123
## [50] 368.5037 366.0085 363.6338 361.3865 359.2703 357.2868 355.4357
## [57] 353.7149 352.1211 350.6494 349.2940 348.0485 346.9060 345.8593
## [64] 344.9001 344.0244 343.2248 342.4951 341.8260 341.2125 340.6502
## [71] 340.1318 339.6603 339.2201 338.8112 338.4312 338.0788 337.7415
## [78] 337.4218 337.1189 336.8242 336.5404 336.2605 335.9839 335.7112
## [85] 335.4362 335.1616 334.8840 334.6037 334.3188 334.0301 333.7371
## [92] 333.4386 333.1375 332.8327 332.5254 332.2168 331.9072 331.5986
## [99] 331.2926
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 331.2926
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 357.2868
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
## (Intercept)  110.6899329
## AtBat         -1.3941293
## Hits           5.3222676
## HmRun          .        
## Runs           .        
## RBI            .        
## Walks          4.4559020
## Years         -8.7478268
## CAtBat         .        
## CHits          .        
## CHmRun         0.4476611
## CRuns          0.6091444
## CRBI           0.3987243
## CWalks        -0.4665820
## LeagueN       31.1402195
## DivisionW   -119.1992215
## PutOuts        0.2682614
## Assists        0.1444617
## Errors        -1.8154257
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 174.4098
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
##  [1] 449.3847 440.2454 431.1471 422.9069 415.4424 406.9532 398.5509
##  [8] 391.3022 385.1212 379.8613 375.3282 371.3965 368.2230 365.6369
## [15] 363.0231 360.0129 357.3996 355.0602 353.0046 351.2700 349.8127
## [22] 348.5271 347.4440 346.5896 346.0065 345.5506 345.1952 344.9491
## [29] 344.8303 344.8022 344.8099 344.8595 345.0133 345.4651 345.9632
## [36] 346.6030 347.3927 348.0270 348.2037 347.9225 347.0604 346.2496
## [43] 345.4943 344.6078 343.8611 343.3973 343.0426 342.7799 342.7929
## [50] 342.9163 343.0866 343.3113 343.5817 343.8021 344.0627 344.4237
## [57] 344.7554 345.0097 345.2111 345.3548 345.4602 345.4878 345.4608
## [64] 345.4901 345.5185 345.6373 345.7032 345.8753 345.9130 346.0712
## [71] 346.1388 346.2882 346.3131 346.3996 346.4846
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.7799
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.3965
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
## 1  255.2820965 201946.6  36438.67  238385.3 165507.91     0
## 2  232.6035386 193816.0  36807.92  230624.0 157008.12     1
## 3  211.9396813 185887.8  36101.22  221989.0 149786.57     2
## 4  193.1115442 178850.3  35089.85  213940.1 143760.42     2
## 5  175.9560468 172592.4  34227.61  206820.0 138364.74     3
## 6  160.3245966 165610.9  33335.63  198946.5 132275.27     4
## 7  146.0818013 158842.9  32271.81  191114.7 126571.05     4
## 8  133.1042967 153117.4  31325.06  184442.5 121792.37     4
## 9  121.2796778 148318.3  30530.83  178849.2 117787.50     4
## 10 110.5055255 144294.6  29802.22  174096.8 114492.37     4
## 11 100.6885192 140871.3  29130.98  170002.2 111740.28     5
## 12  91.7436287 137935.4  28554.00  166489.4 109381.38     5
## 13  83.5933775 135588.2  28051.55  163639.7 107536.65     5
## 14  76.1671723 133690.3  27623.35  161313.7 106066.96     5
## 15  69.4006906 131785.7  27177.84  158963.6 104607.91     6
## 16  63.2353245 129609.3  26679.59  156288.9 102929.70     6
## 17  57.6176726 127734.5  26236.59  153971.1 101497.92     6
## 18  52.4990774 126067.7  25797.28  151865.0 100270.47     6
## 19  47.8352040 124612.2  25380.19  149992.4  99232.03     6
## 20  43.5856563 123390.6  25018.73  148409.3  98371.89     6
## 21  39.7136268 122369.0  24715.20  147084.2  97653.75     6
## 22  36.1855776 121471.2  24475.06  145946.2  96996.10     6
## 23  32.9709506 120717.4  24276.74  144994.1  96440.63     6
## 24  30.0419022 120124.4  24115.31  144239.7  96009.06     6
## 25  27.3730624 119720.5  23974.94  143695.5  95745.59     6
## 26  24.9413150 119405.2  23853.44  143258.6  95551.75     6
## 27  22.7255973 119159.7  23752.60  142912.3  95407.10     6
## 28  20.7067179 118989.9  23668.92  142658.8  95320.96     6
## 29  18.8671902 118907.9  23586.40  142494.3  95321.52     6
## 30  17.1910810 118888.5  23506.60  142395.1  95381.94     7
## 31  15.6638727 118893.9  23440.41  142334.3  95453.49     7
## 32  14.2723374 118928.1  23385.50  142313.6  95542.60     7
## 33  13.0044223 119034.2  23325.73  142359.9  95708.46     9
## 34  11.8491453 119346.1  23232.20  142578.3  96113.94     9
## 35  10.7964999 119690.5  23125.77  142816.3  96564.77     9
## 36   9.8373686 120133.6  23036.59  143170.2  97097.02     9
## 37   8.9634439 120681.7  22960.54  143642.2  97721.12     9
## 38   8.1671562 121122.8  22904.46  144027.3  98218.37    11
## 39   7.4416086 121245.8  22857.26  144103.1  98388.57    11
## 40   6.7805166 121050.1  22724.10  143774.2  98325.99    12
## 41   6.1781542 120450.9  22488.81  142939.7  97962.13    12
## 42   5.6293040 119888.8  22276.59  142165.4  97612.21    13
## 43   5.1292121 119366.3  22088.08  141454.4  97278.22    13
## 44   4.6735471 118754.5  21930.97  140685.5  96823.57    13
## 45   4.2583620 118240.4  21798.77  140039.2  96441.65    13
## 46   3.8800609 117921.7  21692.00  139613.7  96229.73    13
## 47   3.5353670 117678.2  21606.12  139284.4  96072.12    13
## 48   3.2212947 117498.0  21540.42  139038.5  95957.61    13
## 49   2.9351238 117506.9  21518.90  139025.8  95988.04    13
## 50   2.6743755 117591.6  21522.65  139114.2  96068.92    13
## 51   2.4367913 117708.4  21524.09  139232.5  96184.29    13
## 52   2.2203135 117862.6  21509.99  139372.6  96352.66    14
## 53   2.0230670 118048.4  21487.38  139535.8  96561.01    15
## 54   1.8433433 118199.9  21454.42  139654.3  96745.47    15
## 55   1.6795857 118379.1  21449.55  139828.7  96929.59    17
## 56   1.5303760 118627.7  21464.97  140092.7  97162.71    17
## 57   1.3944216 118856.3  21483.87  140340.2  97372.43    17
## 58   1.2705450 119031.7  21497.03  140528.8  97534.69    17
## 59   1.1576733 119170.7  21510.80  140681.5  97659.87    17
## 60   1.0548288 119269.9  21514.62  140784.5  97755.31    17
## 61   0.9611207 119342.7  21512.11  140854.8  97830.62    17
## 62   0.8757374 119361.8  21494.81  140856.6  97867.00    17
## 63   0.7979393 119343.1  21432.80  140775.9  97910.33    17
## 64   0.7270526 119363.4  21400.09  140763.5  97963.29    17
## 65   0.6624632 119383.0  21363.10  140746.1  98019.95    18
## 66   0.6036118 119465.1  21343.95  140809.1  98121.19    18
## 67   0.5499886 119510.7  21314.87  140825.6  98195.86    18
## 68   0.5011291 119629.7  21310.08  140939.8  98319.64    17
## 69   0.4566102 119655.8  21285.41  140941.2  98370.36    18
## 70   0.4160462 119765.3  21278.14  141043.4  98487.14    18
## 71   0.3790858 119812.0  21259.22  141071.3  98552.83    18
## 72   0.3454089 119915.5  21255.28  141170.8  98660.21    18
## 73   0.3147237 119932.8  21232.31  141165.1  98700.46    18
## 74   0.2867645 119992.7  21213.35  141206.0  98779.32    18
## 75   0.2612891 120051.6  21215.93  141267.5  98835.64    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   3.221295   91.74363
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
