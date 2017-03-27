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
##  [1] 450.6510 448.8917 448.2121 447.9444 447.6518 447.3321 446.9829
##  [8] 446.6017 446.1857 445.7321 445.2376 444.6991 444.1130 443.4755
## [15] 442.7828 442.0307 441.2151 440.3316 439.3756 438.3426 437.2279
## [22] 436.0270 434.7352 433.3484 431.8623 430.2733 428.5781 426.7740
## [29] 424.8592 422.8326 420.6942 418.4451 416.0879 413.6264 411.0661
## [36] 408.4142 405.6795 402.8720 400.0038 397.0885 394.1408 391.1766
## [43] 388.2127 385.2661 382.3541 379.4938 376.7013 373.9921 371.3798
## [50] 368.8765 366.4916 364.2343 362.1102 360.1230 358.2744 356.5644
## [57] 354.9908 353.5502 352.2379 351.0481 349.9744 349.0099 348.1472
## [64] 347.3782 346.6961 346.0998 345.5772 345.1197 344.7215 344.3792
## [71] 344.0912 343.8439 343.6383 343.4662 343.3248 343.2087 343.1186
## [78] 343.0417 342.9817 342.9338 342.8935 342.8591 342.8260 342.7956
## [85] 342.7644 342.7296 342.6923 342.6502 342.6013 342.5447 342.4823
## [92] 342.4137 342.3393 342.2572 342.1686 342.0756 341.9783 341.8780
## [99] 341.7760
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 341.776
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 366.4916
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
##                       1
## (Intercept)  24.8990715
## AtBat         .        
## Hits          1.8487366
## HmRun         .        
## Runs          .        
## RBI           .        
## Walks         2.1945815
## Years         .        
## CAtBat        .        
## CHits         .        
## CHmRun        .        
## CRuns         0.2059445
## CRBI          0.4092047
## CWalks        .        
## LeagueN       .        
## DivisionW   -99.7364356
## PutOuts       0.2155667
## Assists       .        
## Errors        .        
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 104.6105
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
##  [1] 449.8281 440.3384 429.9335 421.0525 412.9764 404.8962 396.6847
##  [8] 388.7858 382.2331 376.8130 372.2753 368.2876 364.8617 361.8694
## [15] 359.1330 356.6677 354.4450 352.3535 350.2522 348.4351 346.9535
## [22] 345.7332 344.7299 343.9087 343.2785 342.8215 342.4726 342.2599
## [29] 342.2188 342.2702 342.3407 342.3987 342.4171 342.4157 342.6899
## [36] 343.3268 343.9241 344.4642 344.9379 345.0541 344.9523 344.8180
## [43] 344.5393 343.8930 343.4059 343.1375 342.9261 342.7680 342.8244
## [50] 342.7862 343.1166 343.7734 344.5565 345.2880 345.7841 346.2720
## [57] 346.7343 347.0524 347.2828 347.5192 347.7373 347.8947 348.0314
## [64] 348.2067 348.3491 348.5317 348.6916 348.8403 348.9717 349.0924
## [71] 349.1924
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.2188
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 368.2876
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
## 1  255.2820965 202345.3  24868.69  227214.0 177476.61     0
## 2  232.6035386 193897.9  24664.86  218562.8 169233.08     1
## 3  211.9396813 184842.8  23828.42  208671.3 161014.42     2
## 4  193.1115442 177285.2  23067.48  200352.6 154217.69     2
## 5  175.9560468 170549.5  22273.52  192823.0 148276.01     3
## 6  160.3245966 163940.9  21632.22  185573.1 142308.71     4
## 7  146.0818013 157358.8  20979.77  178338.5 136379.00     4
## 8  133.1042967 151154.4  20266.24  171420.7 130888.17     4
## 9  121.2796778 146102.1  19787.80  165889.9 126314.32     4
## 10 110.5055255 141988.1  19469.09  161457.1 122518.96     4
## 11 100.6885192 138588.9  19258.83  157847.7 119330.04     5
## 12  91.7436287 135635.7  19181.44  154817.2 116454.28     5
## 13  83.5933775 133124.1  19137.38  152261.4 113986.67     5
## 14  76.1671723 130949.4  19109.96  150059.4 111839.47     5
## 15  69.4006906 128976.5  19148.14  148124.7 109828.37     6
## 16  63.2353245 127211.9  19214.53  146426.4 107997.33     6
## 17  57.6176726 125631.2  19295.80  144927.0 106335.44     6
## 18  52.4990774 124153.0  19281.52  143434.5 104871.49     6
## 19  47.8352040 122676.6  19207.49  141884.1 103469.11     6
## 20  43.5856563 121407.0  19149.38  140556.4 102257.63     6
## 21  39.7136268 120376.7  19110.41  139487.2 101266.33     6
## 22  36.1855776 119531.5  19088.01  138619.5 100443.45     6
## 23  32.9709506 118838.7  19081.58  137920.3  99757.15     6
## 24  30.0419022 118273.2  19087.29  137360.5  99185.94     6
## 25  27.3730624 117840.2  19105.72  136945.9  98734.43     6
## 26  24.9413150 117526.6  19134.54  136661.2  98392.07     6
## 27  22.7255973 117287.5  19164.03  136451.5  98123.47     6
## 28  20.7067179 117141.8  19221.58  136363.4  97920.23     6
## 29  18.8671902 117113.7  19326.08  136439.8  97787.60     6
## 30  17.1910810 117148.9  19435.23  136584.1  97713.66     7
## 31  15.6638727 117197.2  19534.30  136731.5  97662.87     7
## 32  14.2723374 117236.9  19616.77  136853.7  97620.12     7
## 33  13.0044223 117249.5  19690.43  136939.9  97559.04     9
## 34  11.8491453 117248.5  19760.10  137008.6  97488.38     9
## 35  10.7964999 117436.4  19786.16  137222.5  97650.20     9
## 36   9.8373686 117873.3  19756.61  137629.9  98116.66     9
## 37   8.9634439 118283.8  19729.38  138013.2  98554.42     9
## 38   8.1671562 118655.6  19740.41  138396.0  98915.20    11
## 39   7.4416086 118982.1  19726.53  138708.7  99255.61    11
## 40   6.7805166 119062.3  19705.22  138767.6  99357.12    12
## 41   6.1781542 118992.1  19536.53  138528.6  99455.53    12
## 42   5.6293040 118899.4  19292.88  138192.3  99606.56    13
## 43   5.1292121 118707.4  19024.62  137732.0  99682.74    13
## 44   4.6735471 118262.4  18692.58  136955.0  99569.85    13
## 45   4.2583620 117927.6  18381.42  136309.0  99546.19    13
## 46   3.8800609 117743.3  18099.10  135842.4  99644.22    13
## 47   3.5353670 117598.3  17830.69  135429.0  99767.65    13
## 48   3.2212947 117489.9  17585.28  135075.2  99904.62    13
## 49   2.9351238 117528.5  17408.93  134937.5 100119.61    13
## 50   2.6743755 117502.4  17239.42  134741.8 100262.97    13
## 51   2.4367913 117729.0  17178.85  134907.9 100550.15    13
## 52   2.2203135 118180.1  17192.40  135372.5 100987.73    14
## 53   2.0230670 118719.2  17209.40  135928.5 101509.76    15
## 54   1.8433433 119223.8  17240.48  136464.3 101983.35    15
## 55   1.6795857 119566.7  17244.25  136810.9 102322.41    17
## 56   1.5303760 119904.3  17257.64  137161.9 102646.67    17
## 57   1.3944216 120224.7  17267.91  137492.6 102956.77    17
## 58   1.2705450 120445.4  17265.94  137711.3 103179.45    17
## 59   1.1576733 120605.3  17273.49  137878.8 103331.85    17
## 60   1.0548288 120769.6  17285.01  138054.6 103484.57    17
## 61   0.9611207 120921.3  17297.09  138218.3 103624.17    17
## 62   0.8757374 121030.7  17293.01  138323.7 103737.71    17
## 63   0.7979393 121125.9  17279.36  138405.2 103846.51    17
## 64   0.7270526 121247.9  17264.17  138512.0 103983.71    17
## 65   0.6624632 121347.1  17252.44  138599.5 104094.66    18
## 66   0.6036118 121474.3  17244.67  138719.0 104229.67    18
## 67   0.5499886 121585.8  17234.04  138819.9 104351.80    18
## 68   0.5011291 121689.6  17227.10  138916.7 104462.47    17
## 69   0.4566102 121781.2  17220.64  139001.9 104560.58    18
## 70   0.4160462 121865.5  17218.17  139083.7 104647.33    18
## 71   0.3790858 121935.3  17207.07  139142.4 104728.26    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   18.86719   91.74363
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
