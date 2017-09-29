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
## (Intercept) 135.070668357
## AtBat         0.107381506
## Hits          0.485804001
## HmRun         1.316978601
## Runs          0.754119187
## RBI           0.728063748
## Walks         1.003524589
## Years         2.785904101
## CAtBat        0.009262397
## CHits         0.036847056
## CHmRun        0.271010647
## CRuns         0.073844717
## CRBI          0.076561001
## CWalks        0.068988790
## LeagueN       6.708052750
## DivisionW   -33.295635138
## PutOuts       0.076376801
## Assists       0.010174962
## Errors       -0.297742861
## NewLeagueN    5.296302685
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
## (Intercept) 135.070668357
## AtBat         0.107381506
## Hits          0.485804001
## HmRun         1.316978601
## Runs          0.754119187
## RBI           0.728063748
## Walks         1.003524589
## Years         2.785904101
## CAtBat        0.009262397
## CHits         0.036847056
## CHmRun        0.271010647
## CRuns         0.073844717
## CRBI          0.076561001
## CWalks        0.068988790
## LeagueN       6.708052750
## DivisionW   -33.295635138
## PutOuts       0.076376801
## Assists       0.010174962
## Errors       -0.297742861
## NewLeagueN    5.296302685
```

```r
sum(coef(fit_ridge_cv, s = "lambda.1se")[-1] ^ 2) # penalty term for lambda one SE
```

```
## [1] 1193.683
```

```r
#predict(fit_ridge_cv, X, s = "lambda.min")
#predict(fit_ridge_cv, X)
mean((y - predict(fit_ridge_cv, X)) ^ 2) # "train error"
```

```
## [1] 122129.1
```

```r
sqrt(fit_ridge_cv$cvm) # CV-RMSEs
```

```
##  [1] 451.3776 450.2010 449.2662 448.7781 448.4883 448.1716 447.8256
##  [8] 447.4480 447.0358 446.5863 446.0963 445.5626 444.9816 444.3497
## [15] 443.6628 442.9170 442.1081 441.2316 440.2830 439.2578 438.1512
## [22] 436.9587 435.6756 434.2976 432.8204 431.2404 429.5539 427.7583
## [29] 425.8515 423.8323 421.7003 419.4564 417.1029 414.6434 412.0828
## [36] 409.4281 406.6876 403.8709 400.9898 398.0575 395.0882 392.0976
## [43] 389.1020 386.1185 383.1640 380.2555 377.4092 374.6406 371.9639
## [50] 369.3909 366.9317 364.5955 362.3881 360.3143 358.3764 356.5745
## [57] 354.9072 353.3712 351.9624 350.6753 349.5038 348.4411 347.4802
## [64] 346.6121 345.8358 345.1412 344.5185 343.9630 343.4659 343.0302
## [71] 342.6456 342.2977 341.9925 341.7262 341.4831 341.2670 341.0736
## [78] 340.8987 340.7359 340.5849 340.4398 340.2990 340.1633 340.0252
## [85] 339.8853 339.7448 339.5965 339.4442 339.2859 339.1197 338.9468
## [92] 338.7686 338.5842 338.3935 338.1989 338.0021 337.8023 337.6013
## [99] 337.4005
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.4005
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 356.5745
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
coef(fit_lasso_cv, s = "lambda.min")
```

```
## 20 x 1 sparse Matrix of class "dgCMatrix"
##                         1
## (Intercept)  134.48030406
## AtBat         -1.67572220
## Hits           5.94122316
## HmRun          0.04746835
## Runs           .         
## RBI            .         
## Walks          4.95676182
## Years        -10.26657309
## CAtBat         .         
## CHits          .         
## CHmRun         0.56236426
## CRuns          0.70135135
## CRBI           0.38727139
## CWalks        -0.58111548
## LeagueN       32.92255640
## DivisionW   -119.37941356
## PutOuts        0.27580087
## Assists        0.19782326
## Errors        -2.26242857
## NewLeagueN     .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.min")[-1])) # penalty term for lambda minimum
```

```
## [1] 180.1579
```

```r
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
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 3.20123
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 123931.3
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 449.7892 439.8327 429.1296 420.3431 412.3643 403.8023 395.2040
##  [8] 387.6904 381.5006 376.4881 372.2562 368.6283 365.3735 362.5083
## [15] 359.7521 357.2600 354.8802 352.5287 350.4744 348.7710 347.3586
## [22] 346.1884 345.2538 344.5753 344.0606 343.6986 343.4375 343.2918
## [29] 343.2537 343.2754 343.3454 343.4485 343.5530 343.6200 343.7430
## [36] 343.9153 344.1952 344.4400 344.6331 344.1380 342.9991 342.0559
## [43] 341.3106 340.5712 339.8190 339.1815 338.7212 338.3938 338.1489
## [50] 337.9274 337.8341 337.7860 337.8247 337.8340 337.8161 337.8298
## [57] 337.8385 337.8458 337.8476 337.8894 337.9403 338.0037 338.1019
## [64] 338.1843 338.3319 338.4057 338.5260 338.6260 338.7233 338.8192
## [71] 338.8960 338.9622 339.0262 339.0894 339.1559
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 337.786
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 365.3735
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
## 1  255.2820965 202310.4  27590.68  229901.0 174719.68     0
## 2  232.6035386 193452.8  27093.65  220546.4 166359.11     1
## 3  211.9396813 184152.2  25837.92  209990.1 158314.29     2
## 4  193.1115442 176688.3  24797.48  201485.8 151890.80     2
## 5  175.9560468 170044.3  23966.68  194011.0 146077.67     3
## 6  160.3245966 163056.3  23329.45  186385.7 139726.81     4
## 7  146.0818013 156186.2  22715.57  178901.8 133470.63     4
## 8  133.1042967 150303.9  22154.31  172458.2 128149.56     4
## 9  121.2796778 145542.7  21718.35  167261.0 123824.32     4
## 10 110.5055255 141743.3  21408.53  163151.9 120334.79     4
## 11 100.6885192 138574.6  21185.23  159759.9 117389.41     5
## 12  91.7436287 135886.9  21047.85  156934.7 114839.01     5
## 13  83.5933775 133497.8  20974.28  154472.1 112523.54     5
## 14  76.1671723 131412.3  20930.24  152342.5 110482.06     5
## 15  69.4006906 129421.6  20941.18  150362.8 108480.42     6
## 16  63.2353245 127634.7  20946.74  148581.5 106687.99     6
## 17  57.6176726 125939.9  20942.49  146882.4 104997.46     6
## 18  52.4990774 124276.5  20914.90  145191.4 103361.61     6
## 19  47.8352040 122832.3  20897.64  143729.9 101934.65     6
## 20  43.5856563 121641.2  20903.67  142544.9 100737.55     6
## 21  39.7136268 120658.0  20927.09  141585.1  99730.94     6
## 22  36.1855776 119846.4  20962.81  140809.2  98883.60     6
## 23  32.9709506 119200.2  21010.81  140211.0  98189.37     6
## 24  30.0419022 118732.2  21074.18  139806.3  97657.97     6
## 25  27.3730624 118377.7  21144.05  139521.8  97233.67     6
## 26  24.9413150 118128.7  21204.82  139333.5  96923.91     6
## 27  22.7255973 117949.3  21265.53  139214.8  96683.79     6
## 28  20.7067179 117849.2  21345.72  139195.0  96503.52     6
## 29  18.8671902 117823.1  21433.42  139256.5  96389.69     6
## 30  17.1910810 117838.0  21519.12  139357.1  96318.85     7
## 31  15.6638727 117886.1  21605.20  139491.3  96280.86     7
## 32  14.2723374 117956.9  21686.22  139643.1  96270.63     7
## 33  13.0044223 118028.6  21759.67  139788.3  96268.97     9
## 34  11.8491453 118074.7  21812.84  139887.6  96261.88     9
## 35  10.7964999 118159.2  21854.81  140014.1  96304.43     9
## 36   9.8373686 118277.7  21886.99  140164.7  96390.74     9
## 37   8.9634439 118470.3  21901.28  140371.6  96569.05     9
## 38   8.1671562 118638.9  21882.75  140521.6  96756.14    11
## 39   7.4416086 118772.0  21830.85  140602.8  96941.12    11
## 40   6.7805166 118431.0  21786.71  140217.7  96644.27    12
## 41   6.1781542 117648.4  21668.61  139317.0  95979.78    12
## 42   5.6293040 117002.2  21586.29  138588.5  95415.96    13
## 43   5.1292121 116492.9  21558.72  138051.6  94934.19    13
## 44   4.6735471 115988.8  21547.22  137536.0  94441.55    13
## 45   4.2583620 115477.0  21546.53  137023.5  93930.42    13
## 46   3.8800609 115044.1  21558.11  136602.2  93485.97    13
## 47   3.5353670 114732.1  21596.48  136328.5  93135.59    13
## 48   3.2212947 114510.4  21638.27  136148.7  92872.11    13
## 49   2.9351238 114344.7  21628.36  135973.1  92716.35    13
## 50   2.6743755 114194.9  21591.17  135786.1  92603.78    13
## 51   2.4367913 114131.9  21559.91  135691.8  92571.97    13
## 52   2.2203135 114099.4  21535.49  135634.9  92563.92    14
## 53   2.0230670 114125.5  21514.22  135639.8  92611.32    15
## 54   1.8433433 114131.8  21494.62  135626.4  92637.18    15
## 55   1.6795857 114119.7  21486.25  135606.0  92633.49    17
## 56   1.5303760 114129.0  21473.20  135602.2  92655.81    17
## 57   1.3944216 114134.9  21454.18  135589.0  92680.68    17
## 58   1.2705450 114139.8  21438.01  135577.8  92701.78    17
## 59   1.1576733 114141.0  21426.34  135567.3  92714.64    17
## 60   1.0548288 114169.2  21417.49  135586.7  92751.75    17
## 61   0.9611207 114203.6  21412.90  135616.5  92790.72    17
## 62   0.8757374 114246.5  21413.86  135660.3  92832.62    17
## 63   0.7979393 114312.9  21415.28  135728.2  92897.60    17
## 64   0.7270526 114368.6  21416.03  135784.6  92952.57    17
## 65   0.6624632 114468.5  21415.27  135883.7  93053.18    18
## 66   0.6036118 114518.4  21402.40  135920.8  93116.01    18
## 67   0.5499886 114599.8  21391.29  135991.1  93208.55    18
## 68   0.5011291 114667.5  21382.77  136050.3  93284.77    17
## 69   0.4566102 114733.4  21370.80  136104.2  93362.65    18
## 70   0.4160462 114798.5  21370.28  136168.7  93428.18    18
## 71   0.3790858 114850.5  21366.24  136216.8  93484.28    18
## 72   0.3454089 114895.4  21361.13  136256.5  93534.22    18
## 73   0.3147237 114938.8  21354.58  136293.4  93584.21    18
## 74   0.2867645 114981.6  21355.90  136337.5  93625.70    18
## 75   0.2612891 115026.7  21360.31  136387.0  93666.40    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.220313   83.59338
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
