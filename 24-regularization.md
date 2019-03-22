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
## (Intercept) 199.418113624
## AtBat         0.093426871
## Hits          0.389767263
## HmRun         1.212875007
## Runs          0.623229048
## RBI           0.618547529
## Walks         0.810467707
## Years         2.544170910
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295031
## DivisionW   -23.545192292
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203267
## NewLeagueN    3.313773161
```


```r
# fitted coefficients, using minimum lambda
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)   71.78758387
## AtBat         -0.58269657
## Hits           2.51715271
## HmRun         -1.39973429
## Runs           1.07259572
## RBI            0.74825248
## Walks          3.17950552
## Years         -8.35976896
## CAtBat         0.00133718
## CHits          0.12772556
## CHmRun         0.68074413
## CRuns          0.27080732
## CRBI           0.24581306
## CWalks        -0.24120197
## LeagueN       51.41107137
## DivisionW   -121.93563374
## PutOuts        0.26073685
## Assists        0.15595798
## Errors        -3.59749877
## NewLeagueN   -15.89754176
```


```r
# penalty term using minimum lambda
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 17868.18
```


```r
# fitted coefficients, using 1-SE rule lambda
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 199.418113624
## AtBat         0.093426871
## Hits          0.389767263
## HmRun         1.212875007
## Runs          0.623229048
## RBI           0.618547529
## Walks         0.810467707
## Years         2.544170910
## CAtBat        0.007897059
## CHits         0.030554662
## CHmRun        0.226545984
## CRuns         0.061265846
## CRBI          0.063384832
## CWalks        0.060720300
## LeagueN       3.743295031
## DivisionW   -23.545192292
## PutOuts       0.056202373
## Assists       0.007879196
## Errors       -0.164203267
## NewLeagueN    3.313773161
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
##  [1] 451.4726 450.4970 449.2488 448.9379 448.6549 448.3457 448.0080
##  [8] 447.6393 447.2370 446.7982 446.3201 445.7993 445.2324 444.6160
## [15] 443.9461 443.2189 442.4302 441.5758 440.6515 439.6527 438.5749
## [22] 437.4138 436.1649 434.8242 433.3875 431.8514 430.2126 428.4686
## [29] 426.6176 424.6586 422.5914 420.4172 418.1383 415.7585 413.2829
## [36] 410.7183 408.0731 405.3570 402.5813 399.7590 396.9039 394.0313
## [43] 391.1570 388.2970 385.4679 382.6855 379.9653 377.3217 374.7676
## [50] 372.3142 369.9712 367.7463 365.6439 363.6693 361.8234 360.1060
## [57] 358.5152 357.0480 355.6998 354.4653 353.3386 352.3132 351.3824
## [64] 350.5387 349.7799 349.0961 348.4798 347.9259 347.4278 346.9832
## [71] 346.5883 346.2353 345.9163 345.6322 345.3754 345.1451 344.9358
## [78] 344.7454 344.5669 344.4006 344.2439 344.0910 343.9451 343.7996
## [85] 343.6535 343.5058 343.3551 343.2007 343.0412 342.8762 342.7061
## [92] 342.5298 342.3489 342.1630 341.9721 341.7797 341.5838 341.3879
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min])
```

```
## [1] 341.3879
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) 
```

```
## [1] 369.9712
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
## (Intercept) 127.95694771
## AtBat         .         
## Hits          1.42342566
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.58214110
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.16027975
## CRBI          0.33667715
## CWalks        .         
## LeagueN       .         
## DivisionW    -8.06171247
## PutOuts       0.08393604
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
## (Intercept)  129.4155569
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
## CWalks        -0.5560143
## LeagueN       32.4646094
## DivisionW   -119.3480842
## PutOuts        0.2741895
## Assists        0.1855978
## Errors        -2.1650837
## NewLeagueN     .
```


```r
# penalty term using minimum lambda
sum(coef(fit_lasso_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 15463.18
```


```r
# fitted coefficients, using 1-SE rule lambda
coef(fit_lasso_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                        1
## (Intercept) 127.95694771
## AtBat         .         
## Hits          1.42342566
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.58214110
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.16027975
## CRBI          0.33667715
## CWalks        .         
## LeagueN       .         
## DivisionW    -8.06171247
## PutOuts       0.08393604
## Assists       .         
## Errors        .         
## NewLeagueN    .
```


```r
# penalty term using 1-SE rule lambda
sum(coef(fit_lasso_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 69.66661
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
## [1] 118581.5
```


```r
# CV-RMSEs
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 449.6773 442.0793 432.7563 424.7044 417.6711 409.5011 400.9324
##  [8] 393.3006 386.9324 381.5628 377.0963 373.1668 369.4551 366.0554
## [15] 362.7563 359.6073 356.9083 354.5875 352.4707 350.6691 349.1712
## [22] 347.9055 346.8680 346.0012 345.2782 344.7054 344.3634 344.1585
## [29] 344.0148 343.9393 343.9118 343.9013 343.8980 343.8877 344.1011
## [36] 344.2544 344.5731 344.9342 345.1250 345.0921 344.7761 344.1212
## [43] 343.2748 342.4468 341.7266 341.0340 340.4715 340.0364 339.6773
## [50] 339.3754 339.1850 339.2421 339.2537 339.3079 339.4391 339.5808
## [57] 339.7203 339.8262 339.9238 340.0548 340.2017 340.3589 340.4950
## [64] 340.6355 340.7892 340.9436 341.1064 341.2581 341.4004 341.5379
## [71] 341.6885 341.8076 341.9221 342.0379 342.1383
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min])
```

```
## [1] 339.185
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) 
```

```
## [1] 362.7563
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
## # A tibble: 75 x 6
##    lambda estimate std.error conf.low conf.high nzero
##     <dbl>    <dbl>     <dbl>    <dbl>     <dbl> <int>
##  1   255.  202210.    33821.  168389.   236031.     0
##  2   233.  195434.    34399.  161035.   229833.     1
##  3   212.  187278.    33400.  153878.   220678.     2
##  4   193.  180374.    32239.  148135.   212613.     2
##  5   176.  174449.    31227.  143222.   205676.     3
##  6   160.  167691.    30228.  137463.   197920.     4
##  7   146.  160747.    29055.  131692.   189802.     4
##  8   133.  154685.    27997.  126688.   182682.     4
##  9   121.  149717.    27088.  122629.   176804.     4
## 10   111.  145590.    26303.  119287.   171893.     4
## # ... with 65 more rows
```

```r
# the two lambda values of interest
glance(fit_lasso_cv) 
```

```
## # A tibble: 1 x 2
##   lambda.min lambda.1se
##        <dbl>      <dbl>
## 1       2.44       69.4
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

The `rmarkdown` file for this chapter can be found [**here**](24-regularization.Rmd). The file was created using `R` version 3.5.2. The following packages (and their dependencies) were loaded when knitting this file:


```
## [1] "caret"   "ggplot2" "lattice" "broom"   "glmnet"  "foreach" "Matrix"
```
