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
# penalty term using 1-SE rule lambda
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 1041.85
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
## [1] 123586
```


```r
# CV-RMSEs
sqrt(fit_ridge_cv$cvm)
```

```
##  [1] 451.1644 449.4658 448.6753 448.4075 448.1148 447.7950 447.4456
##  [8] 447.0642 446.6480 446.1941 445.6993 445.1604 444.5738 443.9358
## [15] 443.2425 442.4896 441.6732 440.7886 439.8315 438.7970 437.6807
## [22] 436.4778 435.1838 433.7943 432.3051 430.7125 429.0131 427.2041
## [29] 425.2836 423.2505 421.1045 418.8468 416.4796 414.0068 411.4336
## [36] 408.7670 406.0157 403.1897 400.3007 397.3622 394.3889 391.3965
## [43] 388.4015 385.4210 382.4723 379.5722 376.7372 373.9823 371.3217
## [50] 368.7672 366.3284 364.0144 361.8315 359.7833 357.8717 356.0967
## [57] 354.4565 352.9478 351.5662 350.3059 349.1608 348.1240 347.1886
## [64] 346.3470 345.5930 344.9235 344.3266 343.7961 343.3258 342.9133
## [71] 342.5538 342.2401 341.9638 341.7256 341.5208 341.3413 341.1881
## [78] 341.0541 340.9374 340.8352 340.7448 340.6613 340.5851 340.5128
## [85] 340.4430 340.3724 340.3016 340.2306 340.1543 340.0751 339.9929
## [92] 339.9052 339.8145 339.7158 339.6184 339.5135 339.4079 339.2981
## [99] 339.1892
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min])
```

```
## [1] 339.1892
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) 
```

```
## [1] 357.8717
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
# fitted coefficients, using minimum lambda
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
# penalty term using minimum lambda
sum(coef(fit_lasso_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 9955.847
```


```r
# fitted coefficients, using 1-SE rule lambda
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
# penalty term using 1-SE rule lambda
sum(coef(fit_lasso_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 3.255212
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
## [1] 127112.4
```


```r
# CV-RMSEs
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 451.0149 441.5669 431.5886 423.2937 415.8737 407.9244 399.7568
##  [8] 392.5482 386.3227 381.1376 376.6615 372.2733 368.2220 364.7332
## [15] 361.7776 359.2282 356.8709 354.6935 352.6353 350.9311 349.5291
## [22] 348.3882 347.4606 346.7103 346.1419 345.7306 345.4458 345.2805
## [29] 345.2175 345.2648 345.3393 345.4476 345.5347 345.6050 345.6327
## [36] 345.9033 346.3021 346.6591 346.9628 347.1529 347.0842 346.5065
## [43] 346.1075 345.8860 345.7704 345.8053 345.8807 345.9843 346.2155
## [50] 346.3758 346.6269 346.9709 347.3463 347.7246 348.0601 348.4574
## [57] 348.8922 349.3759 349.8276 350.2516 350.6660 350.9876 351.2754
## [64] 351.5312 351.7628 351.9540 352.1110 352.2690 352.4162 352.6310
## [71] 352.7647 352.9252 353.0113 353.1120 353.2377 353.3709
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min])
```

```
## [1] 345.2175
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) 
```

```
## [1] 372.2733
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
## 1  255.2820965 203414.5  28813.09  232227.6 174601.40     0
## 2  232.6035386 194981.3  28671.22  223652.6 166310.12     1
## 3  211.9396813 186268.7  27644.14  213912.9 158624.59     2
## 4  193.1115442 179177.6  26804.54  205982.1 152373.03     2
## 5  175.9560468 172950.9  26170.20  199121.1 146780.69     3
## 6  160.3245966 166402.3  25534.55  191936.9 140867.79     4
## 7  146.0818013 159805.5  24750.76  184556.3 135054.74     4
## 8  133.1042967 154094.1  24068.54  178162.7 130025.58     4
## 9  121.2796778 149245.2  23457.77  172703.0 125787.46     4
## 10 110.5055255 145265.9  23002.66  168268.6 122263.25     4
## 11 100.6885192 141873.9  22687.02  164560.9 119186.87     5
## 12  91.7436287 138587.4  22436.28  161023.7 116151.11     5
## 13  83.5933775 135587.4  22192.48  157779.9 113394.95     5
## 14  76.1671723 133030.3  22043.90  155074.2 110986.44     5
## 15  69.4006906 130883.1  21989.30  152872.4 108893.75     6
## 16  63.2353245 129044.9  22009.84  151054.7 107035.04     6
## 17  57.6176726 127356.8  21969.07  149325.9 105387.76     6
## 18  52.4990774 125807.4  21882.78  147690.2 103924.66     6
## 19  47.8352040 124351.6  21767.06  146118.7 102584.59     6
## 20  43.5856563 123152.7  21685.81  144838.5 101466.84     6
## 21  39.7136268 122170.6  21634.84  143805.5 100535.78     6
## 22  36.1855776 121374.4  21603.99  142978.4  99770.38     6
## 23  32.9709506 120728.9  21590.70  142319.6  99138.16     6
## 24  30.0419022 120208.1  21588.40  141796.5  98619.66     6
## 25  27.3730624 119814.2  21592.50  141406.7  98221.75     6
## 26  24.9413150 119529.7  21600.02  141129.7  97929.63     6
## 27  22.7255973 119332.8  21613.34  140946.1  97719.42     6
## 28  20.7067179 119218.6  21631.75  140850.4  97586.86     6
## 29  18.8671902 119175.1  21645.05  140820.2  97530.10     6
## 30  17.1910810 119207.8  21657.82  140865.6  97549.97     7
## 31  15.6638727 119259.3  21670.41  140929.7  97588.85     7
## 32  14.2723374 119334.0  21690.26  141024.3  97643.78     7
## 33  13.0044223 119394.2  21710.22  141104.4  97683.99     9
## 34  11.8491453 119442.8  21721.58  141164.4  97721.24     9
## 35  10.7964999 119461.9  21719.23  141181.2  97742.72     9
## 36   9.8373686 119649.1  21732.76  141381.8  97916.32     9
## 37   8.9634439 119925.2  21760.31  141685.5  98164.85     9
## 38   8.1671562 120172.5  21794.05  141966.6  98378.47    11
## 39   7.4416086 120383.2  21817.10  142200.3  98566.09    11
## 40   6.7805166 120515.1  21815.97  142331.1  98699.18    12
## 41   6.1781542 120467.4  21751.01  142218.4  98716.42    12
## 42   5.6293040 120066.8  21620.12  141686.9  98446.65    13
## 43   5.1292121 119790.4  21519.17  141309.6  98271.26    13
## 44   4.6735471 119637.1  21429.18  141066.3  98207.94    13
## 45   4.2583620 119557.2  21370.26  140927.4  98186.90    13
## 46   3.8800609 119581.3  21368.18  140949.5  98213.15    13
## 47   3.5353670 119633.5  21357.63  140991.1  98275.82    13
## 48   3.2212947 119705.2  21334.69  141039.8  98370.46    13
## 49   2.9351238 119865.2  21317.09  141182.3  98548.12    13
## 50   2.6743755 119976.2  21211.44  141187.6  98764.73    13
## 51   2.4367913 120150.2  21098.55  141248.8  99051.68    13
## 52   2.2203135 120388.8  20985.83  141374.6  99402.94    14
## 53   2.0230670 120649.5  20890.54  141540.0  99758.92    15
## 54   1.8433433 120912.4  20822.49  141734.9 100089.89    15
## 55   1.6795857 121145.9  20784.69  141930.5 100361.17    17
## 56   1.5303760 121422.5  20783.66  142206.2 100638.88    17
## 57   1.3944216 121725.7  20804.44  142530.2 100921.31    17
## 58   1.2705450 122063.5  20839.58  142903.1 101223.95    17
## 59   1.1576733 122379.3  20860.64  143240.0 101518.68    17
## 60   1.0548288 122676.2  20861.75  143538.0 101814.47    17
## 61   0.9611207 122966.6  20859.83  143826.5 102106.79    17
## 62   0.8757374 123192.3  20864.52  144056.8 102327.81    17
## 63   0.7979393 123394.4  20866.73  144261.1 102527.69    17
## 64   0.7270526 123574.2  20865.82  144440.0 102708.36    17
## 65   0.6624632 123737.1  20865.11  144602.2 102871.96    18
## 66   0.6036118 123871.6  20867.17  144738.8 103004.43    18
## 67   0.5499886 123982.1  20871.72  144853.8 103110.41    18
## 68   0.5011291 124093.4  20868.38  144961.8 103225.04    17
## 69   0.4566102 124197.2  20869.55  145066.7 103327.64    18
## 70   0.4160462 124348.6  20869.29  145217.9 103479.32    18
## 71   0.3790858 124442.9  20875.38  145318.3 103567.53    18
## 72   0.3454089 124556.2  20880.95  145437.2 103675.28    18
## 73   0.3147237 124617.0  20883.23  145500.2 103733.76    18
## 74   0.2867645 124688.1  20890.65  145578.7 103797.40    18
## 75   0.2612891 124776.9  20892.62  145669.5 103884.26    18
## 76   0.2380769 124871.0  20895.23  145766.2 103975.78    18
```

```r
# the two lambda values of interest
glance(fit_lasso_cv) 
```

```
##   lambda.min lambda.1se
## 1   18.86719   91.74363
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
