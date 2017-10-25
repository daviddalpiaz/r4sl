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
##  [1] 450.5718 448.9326 448.0875 447.8201 447.5277 447.2083 446.8593
##  [8] 446.4784 446.0627 445.6094 445.1153 444.5771 443.9913 443.3542
## [15] 442.6618 441.9101 441.0948 440.2116 439.2559 438.2231 437.1086
## [22] 435.9078 434.6161 433.2291 431.7428 430.1533 428.4574 426.6524
## [29] 424.7363 422.7080 420.5674 418.3156 415.9551 413.4897 410.9246
## [36] 408.2670 405.5256 402.7102 399.8330 396.9072 393.9476 390.9699
## [43] 387.9907 385.0270 382.0959 379.2144 376.3986 373.6636 371.0232
## [50] 368.4890 366.0716 363.7780 361.6152 359.5866 357.6937 355.9363
## [57] 354.3123 352.8184 351.4496 350.2002 349.0637 348.0333 347.1018
## [64] 346.2618 345.5072 344.8314 344.2288 343.6884 343.2071 342.7774
## [71] 342.3993 342.0646 341.7632 341.4990 341.2621 341.0510 340.8605
## [78] 340.6880 340.5305 340.3834 340.2457 340.1140 339.9857 339.8587
## [85] 339.7329 339.6070 339.4768 339.3432 339.2071 339.0646 338.9190
## [92] 338.7680 338.6122 338.4535 338.2893 338.1251 337.9552 337.7878
## [99] 337.6182
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.6182
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 366.0716
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
##  [1] 449.6216 442.7698 432.7775 424.3356 416.6821 407.7476 399.3451
##  [8] 391.7652 385.1458 379.4369 374.3676 369.7711 365.7771 362.3207
## [15] 359.0536 355.6614 352.5574 349.8988 347.6785 345.8244 344.2794
## [22] 342.9918 341.9174 341.1428 340.5485 340.0718 339.7432 339.5559
## [29] 339.5037 339.5817 339.7066 339.8666 340.0277 340.1636 340.3298
## [36] 340.6826 341.1986 341.4915 341.8003 341.9193 341.5205 341.2019
## [43] 340.9026 340.6324 340.3446 339.7906 339.3843 339.0759 338.8505
## [50] 338.7224 338.6674 338.6734 338.7616 338.8511 338.8593 338.8297
## [57] 338.8814 338.9829 339.1243 339.2946 339.4582 339.5735 339.7144
## [64] 339.8604 339.9960 340.1408 340.2832 340.3922 340.5154 340.5748
## [71] 340.6579 340.7087 340.7948 340.8663 340.9200 340.9749
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 338.6674
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 359.0536
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
## 1  255.2820965 202159.6  26160.23  228319.8 175999.37     0
## 2  232.6035386 196045.1  26945.83  222990.9 169099.26     1
## 3  211.9396813 187296.4  25475.29  212771.7 161821.10     2
## 4  193.1115442 180060.7  24058.68  204119.4 156002.02     2
## 5  175.9560468 173624.0  22851.23  196475.2 150772.77     3
## 6  160.3245966 166258.1  21592.83  187850.9 144665.24     4
## 7  146.0818013 159476.5  20405.05  179881.6 139071.46     4
## 8  133.1042967 153480.0  19295.71  172775.7 134184.25     4
## 9  121.2796778 148337.3  18412.75  166750.1 129924.58     4
## 10 110.5055255 143972.4  17715.09  161687.5 126257.29     4
## 11 100.6885192 140151.1  17218.69  157369.8 122932.39     5
## 12  91.7436287 136730.6  16846.89  153577.5 119883.76     5
## 13  83.5933775 133792.9  16555.63  150348.5 117237.26     5
## 14  76.1671723 131276.3  16338.39  147614.7 114937.90     5
## 15  69.4006906 128919.5  16155.95  145075.5 112763.56     6
## 16  63.2353245 126495.1  15995.71  142490.8 110499.35     6
## 17  57.6176726 124296.7  15862.21  140158.9 108434.49     6
## 18  52.4990774 122429.2  15769.44  138198.6 106659.71     6
## 19  47.8352040 120880.3  15707.13  136587.4 105173.18     6
## 20  43.5856563 119594.5  15667.34  135261.9 103927.21     6
## 21  39.7136268 118528.3  15645.49  134173.8 102882.80     6
## 22  36.1855776 117643.4  15637.08  133280.5 102006.33     6
## 23  32.9709506 116907.5  15639.14  132546.6 101268.34     6
## 24  30.0419022 116378.4  15654.06  132032.5 100724.38     6
## 25  27.3730624 115973.3  15675.92  131649.2 100297.36     6
## 26  24.9413150 115648.9  15701.92  131350.8  99946.94     6
## 27  22.7255973 115425.4  15747.84  131173.3  99677.56     6
## 28  20.7067179 115298.2  15808.92  131107.1  99489.27     6
## 29  18.8671902 115262.8  15869.01  131131.8  99393.75     6
## 30  17.1910810 115315.7  15937.36  131253.1  99378.36     7
## 31  15.6638727 115400.6  16008.91  131409.5  99391.69     7
## 32  14.2723374 115509.3  16088.18  131597.5  99421.13     7
## 33  13.0044223 115618.9  16168.89  131787.8  99449.98     9
## 34  11.8491453 115711.3  16247.59  131958.9  99463.71     9
## 35  10.7964999 115824.4  16316.91  132141.3  99507.49     9
## 36   9.8373686 116064.6  16339.30  132403.9  99725.31     9
## 37   8.9634439 116416.5  16342.19  132758.7 100074.31     9
## 38   8.1671562 116616.5  16340.88  132957.4 100275.59    11
## 39   7.4416086 116827.5  16329.03  133156.5 100498.43    11
## 40   6.7805166 116908.8  16302.50  133211.3 100606.33    12
## 41   6.1781542 116636.3  16212.11  132848.4 100424.14    12
## 42   5.6293040 116418.7  16129.59  132548.3 100289.15    13
## 43   5.1292121 116214.6  16078.49  132293.1 100136.10    13
## 44   4.6735471 116030.5  16028.36  132058.8 100002.09    13
## 45   4.2583620 115834.5  15950.46  131784.9  99884.02    13
## 46   3.8800609 115457.7  15809.36  131267.0  99648.32    13
## 47   3.5353670 115181.7  15693.60  130875.3  99488.08    13
## 48   3.2212947 114972.5  15597.89  130570.4  99374.59    13
## 49   2.9351238 114819.7  15517.93  130337.6  99301.73    13
## 50   2.6743755 114732.9  15442.40  130175.3  99290.48    13
## 51   2.4367913 114695.6  15376.51  130072.1  99319.10    13
## 52   2.2203135 114699.7  15322.37  130022.0  99377.28    14
## 53   2.0230670 114759.4  15290.26  130049.7  99469.18    15
## 54   1.8433433 114820.1  15243.28  130063.4  99576.82    15
## 55   1.6795857 114825.7  15159.50  129985.2  99666.16    17
## 56   1.5303760 114805.6  15068.34  129873.9  99737.22    17
## 57   1.3944216 114840.6  14982.93  129823.6  99857.71    17
## 58   1.2705450 114909.4  14907.58  129817.0 100001.83    17
## 59   1.1576733 115005.3  14841.06  129846.3 100164.21    17
## 60   1.0548288 115120.8  14781.05  129901.9 100339.77    17
## 61   0.9611207 115231.9  14733.16  129965.1 100498.74    17
## 62   0.8757374 115310.2  14707.98  130018.2 100602.20    17
## 63   0.7979393 115405.9  14686.86  130092.7 100719.01    17
## 64   0.7270526 115505.1  14664.10  130169.2 100840.96    17
## 65   0.6624632 115597.3  14652.07  130249.4 100945.22    18
## 66   0.6036118 115695.7  14638.35  130334.1 101057.39    18
## 67   0.5499886 115792.6  14631.58  130424.2 101161.07    18
## 68   0.5011291 115866.9  14623.89  130490.8 101242.98    17
## 69   0.4566102 115950.8  14615.94  130566.7 101334.81    18
## 70   0.4160462 115991.2  14612.17  130603.3 101379.00    18
## 71   0.3790858 116047.8  14602.51  130650.3 101445.29    18
## 72   0.3454089 116082.4  14600.80  130683.2 101481.63    18
## 73   0.3147237 116141.1  14598.74  130739.8 101542.33    18
## 74   0.2867645 116189.9  14598.44  130788.3 101591.42    18
## 75   0.2612891 116226.4  14598.00  130824.4 101628.44    18
## 76   0.2380769 116263.9  14597.05  130860.9 101666.85    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   69.40069
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

The RMarkdown file for this chapter can be found [**here**](15-shrink.Rmd). The file was created using `R` version 3.4.2 and the following packages:

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
## [61] "sfsmisc"      "parallel"     "survival"     "yaml"        
## [65] "colorspace"   "knitr"        "bindr"
```
