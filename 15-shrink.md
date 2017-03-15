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
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 507.788
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 132355.6
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.8094 450.6822 449.8032 449.2840 448.9978 448.6850 448.3435
##  [8] 447.9706 447.5637 447.1199 446.6363 446.1096 445.5362 444.9127
## [15] 444.2352 443.4995 442.7018 441.8376 440.9025 439.8921 438.8019
## [22] 437.6272 436.3638 435.0073 433.5537 431.9994 430.3413 428.5765
## [29] 426.7034 424.7208 422.6286 420.4280 418.1213 415.7122 413.2059
## [36] 410.6094 407.9310 405.1805 402.3695 399.5109 396.6190 393.7090
## [43] 390.7970 387.8994 385.0328 382.2135 379.4571 376.7784 374.1908
## [50] 371.7054 369.3321 367.0784 364.9507 362.9528 361.0863 359.3511
## [57] 357.7455 356.2665 354.9096 353.6696 352.5406 351.5162 350.5897
## [64] 349.7529 349.0020 348.3294 347.7310 347.1988 346.7248 346.3058
## [71] 345.9374 345.6158 345.3304 345.0850 344.8709 344.6842 344.5187
## [78] 344.3746 344.2506 344.1352 344.0334 343.9387 343.8481 343.7589
## [85] 343.6699 343.5779 343.4837 343.3826 343.2760 343.1608 343.0361
## [92] 342.9041 342.7629 342.6106 342.4539 342.2817 342.1125 341.9256
## [99] 341.7503
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.7503
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 371.7054
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
##  [1] 450.0468 441.8578 431.3700 422.5318 414.7656 406.3823 397.8564
##  [8] 390.5759 384.5223 379.3946 375.0624 371.2826 367.9347 364.8028
## [15] 361.7461 358.7828 356.0651 353.6988 351.7243 350.1476 348.8618
## [22] 347.8637 347.2524 346.9348 346.7279 346.6211 346.6312 346.7245
## [29] 346.8746 347.0651 347.3107 347.5493 347.8901 348.6490 349.3621
## [36] 349.9949 350.9250 351.6895 351.7993 351.4820 350.8765 350.0252
## [43] 349.0203 348.1005 347.3457 346.7593 346.3134 345.9578 345.7331
## [50] 345.5763 345.6634 345.9189 346.3264 346.7268 346.9925 347.2359
## [57] 347.4844 347.6888 347.8215 347.8789 347.9687 348.0788 348.2027
## [64] 348.3543 348.5280 348.7226 348.9228 349.0949 349.2708 349.4315
## [71] 349.5747 349.6934
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 345.5763
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 375.0624
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
## 1  255.2820965 202542.2  30065.57  232607.7 172476.60     0
## 2  232.6035386 195238.3  30226.59  225464.9 165011.70     1
## 3  211.9396813 186080.1  28969.76  215049.9 157110.36     2
## 4  193.1115442 178533.1  27836.42  206369.6 150696.71     2
## 5  175.9560468 172030.5  26897.69  198928.2 145132.82     3
## 6  160.3245966 165146.5  26115.42  191262.0 139031.11     4
## 7  146.0818013 158289.7  25426.10  183715.8 132863.60     4
## 8  133.1042967 152549.5  24864.59  177414.1 127684.95     4
## 9  121.2796778 147857.4  24461.60  172319.0 123395.80     4
## 10 110.5055255 143940.3  24200.30  168140.6 119739.97     4
## 11 100.6885192 140671.8  24011.61  164683.4 116660.17     5
## 12  91.7436287 137850.8  23901.47  161752.2 113949.30     5
## 13  83.5933775 135376.0  23836.97  159212.9 111538.99     5
## 14  76.1671723 133081.1  23823.14  156904.2 109257.94     5
## 15  69.4006906 130860.3  23863.52  154723.8 106996.75     6
## 16  63.2353245 128725.1  23963.79  152688.9 104761.33     6
## 17  57.6176726 126782.4  24033.11  150815.5 102749.25     6
## 18  52.4990774 125102.9  24031.04  149133.9 101071.83     6
## 19  47.8352040 123710.0  24021.69  147731.6  99688.27     6
## 20  43.5856563 122603.3  24021.63  146625.0  98581.72     6
## 21  39.7136268 121704.6  24033.11  145737.7  97671.44     6
## 22  36.1855776 121009.2  24080.95  145090.1  96928.22     6
## 23  32.9709506 120584.2  24252.10  144836.3  96332.10     6
## 24  30.0419022 120363.8  24504.60  144868.4  95859.18     6
## 25  27.3730624 120220.2  24743.53  144963.8  95476.70     6
## 26  24.9413150 120146.2  24968.42  145114.6  95177.77     6
## 27  22.7255973 120153.2  25178.93  145332.1  94974.23     6
## 28  20.7067179 120217.9  25377.02  145594.9  94840.86     6
## 29  18.8671902 120322.0  25569.52  145891.5  94752.48     6
## 30  17.1910810 120454.2  25755.58  146209.8  94698.61     7
## 31  15.6638727 120624.7  25927.56  146552.3  94697.17     7
## 32  14.2723374 120790.5  26087.93  146878.5  94702.61     7
## 33  13.0044223 121027.5  26225.69  147253.2  94801.84     9
## 34  11.8491453 121556.2  26347.61  147903.8  95208.54     9
## 35  10.7964999 122053.9  26447.78  148501.7  95606.11     9
## 36   9.8373686 122496.4  26546.40  149042.8  95950.00     9
## 37   8.9634439 123148.4  26585.32  149733.7  96563.06     9
## 38   8.1671562 123685.5  26610.40  150295.9  97075.09    11
## 39   7.4416086 123762.8  26587.80  150350.6  97174.97    11
## 40   6.7805166 123539.6  26485.60  150025.2  97054.03    12
## 41   6.1781542 123114.3  26381.81  149496.1  96732.53    12
## 42   5.6293040 122517.7  26234.11  148751.8  96283.55    13
## 43   5.1292121 121815.2  25980.73  147795.9  95834.46    13
## 44   4.6735471 121174.0  25689.30  146863.3  95484.66    13
## 45   4.2583620 120649.0  25407.52  146056.5  95241.51    13
## 46   3.8800609 120242.0  25141.66  145383.7  95100.37    13
## 47   3.5353670 119933.0  24890.02  144823.0  95042.95    13
## 48   3.2212947 119686.8  24664.34  144351.1  95022.43    13
## 49   2.9351238 119531.4  24470.10  144001.5  95061.29    13
## 50   2.6743755 119423.0  24298.58  143721.5  95124.39    13
## 51   2.4367913 119483.2  24134.32  143617.5  95348.85    13
## 52   2.2203135 119659.9  23957.77  143617.7  95702.12    14
## 53   2.0230670 119942.0  23788.54  143730.5  96153.44    15
## 54   1.8433433 120219.4  23622.71  143842.2  96596.73    15
## 55   1.6795857 120403.8  23443.98  143847.8  96959.82    17
## 56   1.5303760 120572.7  23278.21  143851.0  97294.53    17
## 57   1.3944216 120745.4  23154.82  143900.3  97590.62    17
## 58   1.2705450 120887.5  23048.79  143936.3  97838.73    17
## 59   1.1576733 120979.8  22955.52  143935.3  98024.26    17
## 60   1.0548288 121019.7  22873.81  143893.5  98145.91    17
## 61   0.9611207 121082.2  22802.94  143885.1  98279.25    17
## 62   0.8757374 121158.8  22740.88  143899.7  98417.97    17
## 63   0.7979393 121245.1  22686.46  143931.6  98558.68    17
## 64   0.7270526 121350.7  22646.49  143997.2  98704.26    17
## 65   0.6624632 121471.8  22624.36  144096.2  98847.44    18
## 66   0.6036118 121607.4  22609.41  144216.8  98998.04    18
## 67   0.5499886 121747.1  22592.04  144339.2  99155.08    18
## 68   0.5011291 121867.2  22581.89  144449.1  99285.33    17
## 69   0.4566102 121990.1  22568.79  144558.9  99421.31    18
## 70   0.4160462 122102.3  22563.09  144665.4  99539.25    18
## 71   0.3790858 122202.5  22555.54  144758.0  99646.93    18
## 72   0.3454089 122285.5  22548.43  144833.9  99737.03    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.674375   100.6885
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
