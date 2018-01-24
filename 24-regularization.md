# Regularization

**Chapter Status:** Currently this chapter is very sparse. It essentially only expands upon an example discussed in ISL, thus only illustrates usage of the methods. Mathematical and conceptual details of the methods will be added later. Also, more comments on using `glmnet` with `caret` will be discussed.



We will use the `Hitters` dataset from the `ISLR` package to explore two shrinkage methods: **ridge regression** and **lasso**. These are otherwise known as **penalized regression** methods.


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

We use the `glmnet()` and `cv.glmnet()` functions from the `glmnet` package to fit penalized regressions.


```r
library(glmnet)
```

Unfortunately, the `glmnet` function does not allow the use of model formulas, so we setup the data for ease of use with `glmnet`. Eventually we will use `train()` from `caret` which does allow for fitting penalized regression with the formula syntax, but to explore some of the details, we first work with the functions from `glmnet` directly.


```r
X = model.matrix(Salary ~ ., Hitters)[, -1]
y = Hitters$Salary
```

First, we fit an ordinary linear regression, and note the size of the predictors' coefficients, and predictors' coefficients squared. (The two penalties we will use.)


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

Notice that the intercept is **not** penalized. Also, note that that ridge regression is **not** scale invariant like the usual unpenalized regression. Thankfully, `glmnet()` takes care of this internally. It automatically standardizes predictors for fitting, then reports fitted coefficient using the original scale.

The two plots illustrate how much the coefficients are penalized for different values of $\lambda$. Notice none of the coefficients are forced to be zero.


```r
par(mfrow = c(1, 2))
fit_ridge = glmnet(X, y, alpha = 0)
plot(fit_ridge)
plot(fit_ridge, xvar = "lambda", label = TRUE)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/ridge-1} \end{center}

We use cross-validation to select a good $\lambda$ value. The `cv.glmnet()`function uses 10 folds by default. The plot illustrates the MSE for the $\lambda$s considered. Two lines are drawn. The first is the $\lambda$ that gives the smallest MSE. The second is the $\lambda$ that gives an MSE within one standard error of the smallest.


```r
fit_ridge_cv = cv.glmnet(X, y, alpha = 0)
plot(fit_ridge_cv)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-7-1} \end{center}

The `cv.glmnet()` function returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are smaller than the full linear regression. (As we would expect.)


```r
# fitted coefficients, using 1-SE rule lambda, default behavior
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
# fitted coefficients, using minimum lambda
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
# penalty term using minimum lambda
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 18126.85
```


```r
# fitted coefficients, using 1-SE rule lambda
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
# penalty term using 1-SE rule lambda
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 588.9958
```


```r
# predict using minimum lambda
predict(fit_ridge_cv, X, s = "lambda.min")
```


```r
# predict using 1-SE rule lambda, default behavior
predict(fit_ridge_cv, X)
```


```r
# calcualte "train error"
mean((y - predict(fit_ridge_cv, X)) ^ 2)
```

```
## [1] 130404.9
```


```r
# CV-RMSEs
sqrt(fit_ridge_cv$cvm)
```

```
##  [1] 451.9422 450.4961 450.0847 449.3919 449.1032 448.7876 448.4430
##  [8] 448.0668 447.6562 447.2085 446.7205 446.1889 445.6103 444.9810
## [15] 444.2972 443.5547 442.7494 441.8770 440.9330 439.9128 438.8118
## [22] 437.6255 436.3494 434.9791 433.5106 431.9401 430.2644 428.4806
## [29] 426.5869 424.5822 422.4662 420.2401 417.9061 415.4679 412.9307
## [36] 410.3013 407.5883 404.8013 401.9521 399.0537 396.1204 393.1677
## [43] 390.2118 387.2692 384.3569 381.4913 378.6883 375.9627 373.3280
## [50] 370.7960 368.3763 366.0773 363.9047 361.8623 359.9514 358.1717
## [57] 356.5238 355.0008 353.5993 352.3156 351.1424 350.0729 349.1004
## [64] 348.2176 347.4195 346.6977 346.0467 345.4624 344.9338 344.4572
## [71] 344.0320 343.6493 343.3058 342.9963 342.7167 342.4626 342.2366
## [78] 342.0248 341.8338 341.6546 341.4880 341.3302 341.1790 341.0322
## [85] 340.8896 340.7492 340.6080 340.4674 340.3261 340.1841 340.0396
## [92] 339.8944 339.7476 339.6002 339.4529 339.3064 339.1617 339.0185
## [99] 338.8803
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min])
```

```
## [1] 338.8803
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) 
```

```
## [1] 368.3763
```


## Lasso

We now illustrate **lasso**, which can be fit using `glmnet()` with `alpha = 1` and seeks to minimize

$$
\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij}    \right) ^ 2 + \lambda \sum_{j=1}^{p} |\beta_j| .
$$

Like ridge, lasso is not scale invariant.

The two plots illustrate how much the coefficients are penalized for different values of $\lambda$. Notice some of the coefficients are forced to be zero.


```r
par(mfrow = c(1, 2))
fit_lasso = glmnet(X, y, alpha = 1)
plot(fit_lasso)
plot(fit_lasso, xvar = "lambda", label = TRUE)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/lasso-1} \end{center}

Again, to actually pick a $\lambda$, we will use cross-validation. The plot is similar to the ridge plot. Notice along the top is the number of features in the model. (Which changed in this plot.)


```r
fit_lasso_cv = cv.glmnet(X, y, alpha = 1)
plot(fit_lasso_cv)
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-19-1} \end{center}

`cv.glmnet()` returns several details of the fit for both $\lambda$ values in the plot. Notice the penalty terms are again smaller than the full linear regression. (As we would expect.) Some coefficients are 0.


```r
# fitted coefficients, using 1-SE rule lambda, default behavior
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
# fitted coefficients, using minimum lambda
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
# penalty term using minimum lambda
sum(coef(fit_lasso_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 15414.79
```


```r
# fitted coefficients, using 1-SE rule lambda
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
# penalty term using 1-SE rule lambda
sum(coef(fit_lasso_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 3.751855
```


```r
# predict using minimum lambda
predict(fit_lasso_cv, X, s = "lambda.min")
```


```r
# predict using 1-SE rule lambda, default behavior
predict(fit_lasso_cv, X)
```


```r
# calcualte "train error"
mean((y - predict(fit_lasso_cv, X)) ^ 2)
```

```
## [1] 123931.3
```


```r
# CV-RMSEs
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 449.5105 441.8682 431.9440 423.5422 415.7818 406.9109 398.4101
##  [8] 391.0643 384.7088 379.3889 375.0000 371.3537 368.3680 365.9734
## [15] 363.6642 361.3240 359.0428 356.7625 354.7328 353.0299 351.6310
## [22] 350.4809 349.5345 348.7737 348.2604 348.0156 347.8457 347.7580
## [29] 347.7375 347.7618 347.8350 347.8988 347.9235 347.9588 348.1109
## [36] 348.4894 349.0082 349.3901 349.4694 348.9626 347.9144 346.8117
## [43] 345.6787 344.7736 343.9756 343.2604 342.6977 342.2801 342.0334
## [50] 341.9722 341.9884 342.1956 342.3460 342.4439 342.4887 342.5238
## [57] 342.5042 342.5814 342.7384 342.9062 343.1065 343.3161 343.5066
## [64] 343.6588 343.7716 343.9180 344.0456 344.1548 344.2459 344.3619
## [71] 344.4208 344.4893
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min])
```

```
## [1] 341.9722
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) 
```

```
## [1] 368.368
```


## `broom`

Sometimes, the output from `glmnet()` can be overwhelming. The `broom` package can help with that.


```r
library(broom)
# the output from the commented line would be immense
# fit_lasso_cv
tidy(fit_lasso_cv)
```

```
##         lambda estimate std.error conf.high  conf.low nzero
## 1  255.2820965 202059.7  22461.11  224520.8 179598.61     0
## 2  232.6035386 195247.5  22774.61  218022.1 172472.91     1
## 3  211.9396813 186575.6  21751.71  208327.3 164823.92     2
## 4  193.1115442 179388.0  20946.99  200335.0 158441.01     2
## 5  175.9560468 172874.5  20341.58  193216.1 152532.96     3
## 6  160.3245966 165576.4  19873.20  185449.6 145703.25     4
## 7  146.0818013 158730.6  19583.26  178313.9 139147.34     4
## 8  133.1042967 152931.3  19460.91  172392.2 133470.41     4
## 9  121.2796778 148000.8  19472.54  167473.4 128528.29     4
## 10 110.5055255 143935.9  19576.18  163512.1 124359.73     4
## 11 100.6885192 140625.0  19737.61  160362.6 120887.39     5
## 12  91.7436287 137903.6  19960.26  157863.9 117943.33     5
## 13  83.5933775 135695.0  20223.22  155918.2 115471.78     5
## 14  76.1671723 133936.6  20477.53  154414.1 113459.02     5
## 15  69.4006906 132251.7  20633.75  152885.4 111617.91     6
## 16  63.2353245 130555.0  20721.41  151276.4 109833.61     6
## 17  57.6176726 128911.7  20822.66  149734.4 108089.06     6
## 18  52.4990774 127279.5  20859.29  148138.8 106420.21     6
## 19  47.8352040 125835.4  20838.22  146673.6 104997.16     6
## 20  43.5856563 124630.1  20836.18  145466.3 103793.97     6
## 21  39.7136268 123644.4  20854.45  144498.8 102789.92     6
## 22  36.1855776 122836.9  20887.50  143724.4 101949.39     6
## 23  32.9709506 122174.3  20930.85  143105.2 101243.48     6
## 24  30.0419022 121643.1  20982.07  142625.2 100661.05     6
## 25  27.3730624 121285.3  21036.49  142321.8 100248.81     6
## 26  24.9413150 121114.9  21082.14  142197.0 100032.72     6
## 27  22.7255973 120996.7  21128.11  142124.8  99868.55     6
## 28  20.7067179 120935.6  21173.76  142109.4  99761.83     6
## 29  18.8671902 120921.4  21218.10  142139.5  99703.27     6
## 30  17.1910810 120938.3  21260.51  142198.8  99677.77     7
## 31  15.6638727 120989.2  21301.17  142290.3  99688.00     7
## 32  14.2723374 121033.6  21329.48  142363.1  99704.10     7
## 33  13.0044223 121050.8  21351.58  142402.4  99699.21     9
## 34  11.8491453 121075.3  21378.42  142453.7  99696.87     9
## 35  10.7964999 121181.2  21395.24  142576.5  99785.98     9
## 36   9.8373686 121444.9  21421.81  142866.7 100023.06     9
## 37   8.9634439 121806.7  21485.09  143291.8 100321.60     9
## 38   8.1671562 122073.5  21574.69  143648.1 100498.77    11
## 39   7.4416086 122128.9  21650.77  143779.6 100478.11    11
## 40   6.7805166 121774.9  21606.08  143380.9 100168.79    12
## 41   6.1781542 121044.4  21484.78  142529.2  99559.64    12
## 42   5.6293040 120278.4  21371.42  141649.8  98906.96    13
## 43   5.1292121 119493.7  21258.30  140752.0  98235.45    13
## 44   4.6735471 118868.9  21159.58  140028.4  97709.27    13
## 45   4.2583620 118319.2  21081.49  139400.7  97237.74    13
## 46   3.8800609 117827.7  21023.06  138850.8  96804.66    13
## 47   3.5353670 117441.7  20971.36  138413.1  96470.35    13
## 48   3.2212947 117155.7  20924.76  138080.5  96230.93    13
## 49   2.9351238 116986.9  20872.66  137859.5  96114.21    13
## 50   2.6743755 116945.0  20831.42  137776.4  96113.55    13
## 51   2.4367913 116956.1  20808.80  137764.9  96147.29    13
## 52   2.2203135 117097.8  20770.99  137868.8  96326.85    14
## 53   2.0230670 117200.8  20736.67  137937.5  96464.12    15
## 54   1.8433433 117267.8  20717.59  137985.4  96550.22    15
## 55   1.6795857 117298.5  20693.87  137992.4  96604.63    17
## 56   1.5303760 117322.6  20641.39  137963.9  96681.17    17
## 57   1.3944216 117309.1  20580.85  137890.0  96728.30    17
## 58   1.2705450 117362.0  20516.26  137878.3  96845.78    17
## 59   1.1576733 117469.6  20443.28  137912.9  97026.34    17
## 60   1.0548288 117584.7  20376.76  137961.4  97207.91    17
## 61   0.9611207 117722.1  20317.85  138039.9  97404.22    17
## 62   0.8757374 117866.0  20263.08  138129.1  97602.90    17
## 63   0.7979393 117996.8  20214.75  138211.5  97782.02    17
## 64   0.7270526 118101.4  20175.29  138276.7  97926.10    17
## 65   0.6624632 118178.9  20144.11  138323.0  98034.82    18
## 66   0.6036118 118279.6  20114.28  138393.9  98165.30    18
## 67   0.5499886 118367.4  20088.59  138456.0  98278.78    18
## 68   0.5011291 118442.5  20065.47  138508.0  98377.03    17
## 69   0.4566102 118505.3  20045.83  138551.1  98459.42    18
## 70   0.4160462 118585.1  20033.92  138619.0  98551.19    18
## 71   0.3790858 118625.7  20016.86  138642.6  98608.86    18
## 72   0.3454089 118672.9  19999.92  138672.8  98672.93    18
```

```r
# the two lambda values of interest
glance(fit_lasso_cv) 
```

```
##   lambda.min lambda.1se
## 1   2.674375   83.59338
```


## Simulated Data, $p > n$

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



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-34-1} \end{center}


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
par(mfrow = c(1, 2))
plot(glmnet(X, y, family = "binomial"))
plot(glmnet(X, y, family = "binomial"), xvar = "lambda")
```



\begin{center}\includegraphics{24-regularization_files/figure-latex/unnamed-chunk-38-1} \end{center}

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

The interaction between the `glmnet` and `caret` packages is sometimes frustrating, but for obtaining results for particular values of $\lambda$, we see it can be easily used. More on this next chapter.


## External Links

- [`glmnet` Web Vingette](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html) - Details from the package developers.


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](24-regularization.Rmd). The file was created using `R` version 3.4.2. The following packages (and their dependencies) were loaded when knitting this file:


```
## [1] "caret"   "ggplot2" "lattice" "broom"   "glmnet"  "foreach" "Matrix"
```
