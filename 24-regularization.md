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
##  [1] "AtBat"     "Hits"      "HmRun"     "Runs"      "RBI"       "Walks"    
##  [7] "Years"     "CAtBat"    "CHits"     "CHmRun"    "CRuns"     "CRBI"     
## [13] "CWalks"    "League"    "Division"  "PutOuts"   "Assists"   "Errors"   
## [19] "Salary"    "NewLeague"
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
##  (Intercept)        AtBat         Hits        HmRun         Runs          RBI 
##  163.1035878   -1.9798729    7.5007675    4.3308829   -2.3762100   -1.0449620 
##        Walks        Years       CAtBat        CHits       CHmRun        CRuns 
##    6.2312863   -3.4890543   -0.1713405    0.1339910   -0.1728611    1.4543049 
##         CRBI       CWalks      LeagueN    DivisionW      PutOuts      Assists 
##    0.8077088   -0.8115709   62.5994230 -116.8492456    0.2818925    0.3710692 
##       Errors   NewLeagueN 
##   -3.3607605  -24.7623251
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
## (Intercept) 268.287904623
## AtBat         0.075738253
## Hits          0.300154605
## HmRun         1.022784254
## Runs          0.489474364
## RBI           0.495632198
## Walks         0.626356704
## Years         2.143185625
## CAtBat        0.006369369
## CHits         0.024201921
## CHmRun        0.180499284
## CRuns         0.048544437
## CRBI          0.050169414
## CWalks        0.049897906
## LeagueN       1.802540410
## DivisionW   -16.185025086
## PutOuts       0.040146198
## Assists       0.005930000
## Errors       -0.087618226
## NewLeagueN    1.836629069
```


```r
# fitted coefficients, using minimum lambda
coef(fit_ridge_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  8.112693e+01
## AtBat       -6.815959e-01
## Hits         2.772312e+00
## HmRun       -1.365680e+00
## Runs         1.014826e+00
## RBI          7.130225e-01
## Walks        3.378558e+00
## Years       -9.066800e+00
## CAtBat      -1.199478e-03
## CHits        1.361029e-01
## CHmRun       6.979958e-01
## CRuns        2.958896e-01
## CRBI         2.570711e-01
## CWalks      -2.789666e-01
## LeagueN      5.321272e+01
## DivisionW   -1.228345e+02
## PutOuts      2.638876e-01
## Assists      1.698796e-01
## Errors      -3.685645e+00
## NewLeagueN  -1.810510e+01
```


```r
# penalty term using minimum lambda
sum(coef(fit_ridge_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 18367.29
```


```r
# fitted coefficients, using 1-SE rule lambda
coef(fit_ridge_cv, s = "lambda.1se")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept) 268.287904623
## AtBat         0.075738253
## Hits          0.300154605
## HmRun         1.022784254
## Runs          0.489474364
## RBI           0.495632198
## Walks         0.626356704
## Years         2.143185625
## CAtBat        0.006369369
## CHits         0.024201921
## CHmRun        0.180499284
## CRuns         0.048544437
## CRBI          0.050169414
## CWalks        0.049897906
## LeagueN       1.802540410
## DivisionW   -16.185025086
## PutOuts       0.040146198
## Assists       0.005930000
## Errors       -0.087618226
## NewLeagueN    1.836629069
```


```r
# penalty term using 1-SE rule lambda
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2)
```

```
## [1] 275.24
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
## [1] 141009.7
```


```r
# CV-RMSEs
sqrt(fit_ridge_cv$cvm)
```

```
##   [1] 451.6082 449.3727 448.9465 448.6777 448.3838 448.0627 447.7119 447.3290
##   [9] 446.9111 446.4554 445.9586 445.4176 444.8287 444.1882 443.4922 442.7363
##  [17] 441.9167 441.0287 440.0678 439.0294 437.9088 436.7014 435.4025 434.0079
##  [25] 432.5132 430.9149 429.2094 427.3942 425.4671 423.4272 421.2744 419.0097
##  [33] 416.6357 414.1560 411.5763 408.9035 406.1466 403.3156 400.4226 397.4813
##  [41] 394.5065 391.5143 388.5216 385.5456 382.6039 379.7137 376.8916 374.1531
##  [49] 371.5125 368.9820 366.5713 364.2897 362.1435 360.1365 358.2707 356.5461
##  [57] 354.9610 353.5119 352.1943 351.0025 349.9301 348.9702 348.1155 347.3576
##  [65] 346.6927 346.1154 345.6123 345.1796 344.8101 344.5019 344.2503 344.0409
##  [73] 343.8728 343.7479 343.6560 343.5887 343.5488 343.5297 343.5269 343.5377
##  [81] 343.5586 343.5863 343.6179 343.6522 343.6847 343.7140 343.7393 343.7585
##  [89] 343.7687 343.7707 343.7617 343.7422 343.7125 343.6697 343.6172 343.5524
##  [97] 343.4768 343.3919 343.2948 343.2022
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min])
```

```
## [1] 343.2022
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) 
```

```
## [1] 379.7137
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
##                         1
## (Intercept)  134.48030383
## AtBat         -1.67572220
## Hits           5.94122316
## HmRun          0.04746835
## Runs           .         
## RBI            .         
## Walks          4.95676182
## Years        -10.26657307
## CAtBat         .         
## CHits          .         
## CHmRun         0.56236426
## CRuns          0.70135135
## CRBI           0.38727139
## CWalks        -0.58111548
## LeagueN       32.92255638
## DivisionW   -119.37941356
## PutOuts        0.27580087
## Assists        0.19782325
## Errors        -2.26242857
## NewLeagueN     .
```


```r
# penalty term using minimum lambda
sum(coef(fit_lasso_cv, s = "lambda.min")[-1] ^ 2)
```

```
## [1] 15509.95
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
##  [1] 449.7775 439.9842 429.5458 420.7172 412.5926 404.0150 395.7832 388.2937
##  [9] 381.9885 376.6325 372.1613 368.3146 364.9948 362.0247 359.3377 356.3046
## [17] 353.3899 350.9218 348.8665 347.1583 345.7410 344.5636 343.5920 342.7921
## [25] 342.1468 341.6665 341.3152 341.0524 340.9313 340.9155 340.9583 341.0029
## [33] 341.0482 341.1761 341.4915 341.9013 342.4281 342.8232 342.5913 342.0171
## [41] 341.1181 340.1917 339.4381 338.6261 337.7614 337.0948 336.5406 336.0906
## [49] 335.7161 335.3970 335.2516 335.1692 335.1765 335.1828 335.1956 335.1814
## [57] 335.2026 335.3277 335.3455 335.3862 335.4355 335.4875 335.5639 335.6569
## [65] 335.7619 335.8604 335.9750 336.0727 336.1725 336.2484 336.3536 336.4611
## [73] 336.5295 336.6233 336.7015 336.7786 336.8293 336.9081 336.9736 336.9889
```


```r
# CV-RMSE using minimum lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min])
```

```
## [1] 335.1692
```


```r
# CV-RMSE using 1-SE rule lambda
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) 
```

```
## [1] 359.3377
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
## # A tibble: 80 x 6
##    lambda estimate std.error conf.low conf.high nzero
##     <dbl>    <dbl>     <dbl>    <dbl>     <dbl> <int>
##  1   255.  202300.    23338.  178962.   225637.     0
##  2   233.  193586.    23779.  169807.   217365.     1
##  3   212.  184510.    22822.  161687.   207332.     2
##  4   193.  177003.    22023.  154980.   199026.     2
##  5   176.  170233.    21430.  148803.   191663.     3
##  6   160.  163228.    21037.  142191.   184265.     4
##  7   146.  156644.    20382.  136262.   177027.     4
##  8   133.  150772.    19854.  130918.   170626.     4
##  9   121.  145915.    19434.  126481.   165349.     4
## 10   111.  141852.    19031.  122821.   160883.     4
## # ... with 70 more rows
```

```r
# the two lambda values of interest
glance(fit_lasso_cv) 
```

```
## # A tibble: 1 x 3
##   lambda.min lambda.1se  nobs
##        <dbl>      <dbl> <int>
## 1       2.22       69.4   263
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
##  s0  s1  s2  s3  s4  s5  s6  s7  s8  s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 
##   0   2   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 
## s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 s34 s35 s36 s37 s38 s39 
##   3   3   3   3   3   3   3   3   3   3   4   6   7  10  18  24  35  54  65  75 
## s40 s41 s42 s43 s44 s45 s46 s47 s48 s49 s50 s51 s52 s53 s54 s55 s56 s57 s58 s59 
##  86 100 110 129 147 168 187 202 221 241 254 269 283 298 310 324 333 350 364 375 
## s60 s61 s62 s63 s64 s65 s66 s67 s68 s69 s70 s71 s72 s73 s74 s75 s76 s77 s78 s79 
## 387 400 411 429 435 445 453 455 462 466 475 481 487 491 496 498 502 504 512 518 
## s80 s81 s82 s83 s84 s85 s86 s87 s88 s89 s90 s91 s92 s93 s94 s95 s96 s97 s98 s99 
## 523 526 528 536 543 550 559 561 563 566 570 571 576 582 586 590 596 596 600 599
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
## [1] 0.03718493
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
## 1     1 0.03718493
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
## 1     1 0.03718493 0.7560401 0.5119754 0.01738219 0.03459404
## 2     1 0.05149690 0.7690353 0.5380106 0.02166057 0.04323249
```

The interaction between the `glmnet` and `caret` packages is sometimes frustrating, but for obtaining results for particular values of $\lambda$, we see it can be easily used. More on this next chapter.


## External Links

- [`glmnet` Web Vingette](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html) - Details from the package developers.


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](24-regularization.Rmd). The file was created using `R` version 4.0.2. The following packages (and their dependencies) were loaded when knitting this file:


```
## [1] "caret"   "ggplot2" "lattice" "broom"   "glmnet"  "Matrix"
```
