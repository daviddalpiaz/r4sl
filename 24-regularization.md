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
##  [1] 450.5072 449.1569 448.1642 447.9011 447.6135 447.2993 446.9560
##  [8] 446.5813 446.1724 445.7265 445.2405 444.7112 444.1351 443.5085
## [15] 442.8276 442.0884 441.2867 440.4182 439.4786 438.4632 437.3676
## [22] 436.1872 434.9176 433.5545 432.0939 430.5321 428.8660 427.0930
## [29] 425.2111 423.2195 421.1180 418.9078 416.5915 414.1728 411.6572
## [36] 409.0516 406.3649 403.6069 400.7895 397.9261 395.0312 392.1205
## [43] 389.2104 386.3178 383.4598 380.6529 377.9134 375.2560 372.6947
## [50] 370.2409 367.9039 365.6926 363.6128 361.6678 359.8593 358.1870
## [57] 356.6487 355.2409 353.9587 352.7965 351.7475 350.8049 349.9612
## [64] 349.2084 348.5422 347.9541 347.4396 346.9859 346.5897 346.2472
## [71] 345.9515 345.6997 345.4838 345.2988 345.1388 345.0082 344.8965
## [78] 344.7957 344.7137 344.6364 344.5696 344.5068 344.4442 344.3837
## [85] 344.3177 344.2490 344.1787 344.1005 344.0159 343.9230 343.8227
## [92] 343.7150 343.5998 343.4774 343.3483 343.2133 343.0726 342.9270
## [99] 342.7778
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 342.7778
```

```r
sqrt(fit_ridge_cv$cvm[fit_ridge_cv$lambda == fit_ridge_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 367.9039
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
## (Intercept) 144.37970458
## AtBat         .         
## Hits          1.36380384
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.49731098
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.15275165
## CRBI          0.32833941
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.06625755
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
## (Intercept) 144.37970458
## AtBat         .         
## Hits          1.36380384
## HmRun         .         
## Runs          .         
## RBI           .         
## Walks         1.49731098
## Years         .         
## CAtBat        .         
## CHits         .         
## CHmRun        .         
## CRuns         0.15275165
## CRBI          0.32833941
## CWalks        .         
## LeagueN       .         
## DivisionW     .         
## PutOuts       0.06625755
## Assists       .         
## Errors        .         
## NewLeagueN    .
```

```r
sum(abs(coef(fit_lasso_cv, s = "lambda.1se")[-1])) # penalty term for lambda one SE
```

```
## [1] 3.408463
```

```r
#predict(fit_lasso_cv, X, s = "lambda.min")
#predict(fit_lasso_cv, X)
mean((y - predict(fit_lasso_cv, X)) ^ 2) # "train error"
```

```
## [1] 121290.9
```

```r
sqrt(fit_lasso_cv$cvm)
```

```
##  [1] 449.5493 440.3007 430.4708 422.0452 414.3983 406.8503 398.2525
##  [8] 390.7018 384.3629 379.1944 374.9133 371.1192 367.7927 364.8996
## [15] 361.8896 358.6313 355.5839 353.0505 350.9548 349.2250 347.7903
## [22] 346.6099 345.7429 345.1053 344.5945 344.1801 343.8788 343.6458
## [29] 343.4790 343.3218 343.2148 343.1896 343.1946 343.2173 343.3456
## [36] 343.7014 344.1656 344.5651 344.4588 343.8558 343.1405 342.1619
## [43] 340.9537 339.7706 338.8296 338.0456 337.3607 336.8313 336.4394
## [50] 336.2129 336.0759 336.0817 336.1688 336.3532 336.5282 336.6165
## [57] 336.6468 336.7155 336.7485 336.7956 336.9663 337.1938 337.4229
## [64] 337.6476 337.8441 338.0744 338.2448 338.4681 338.6316 338.7920
## [71] 338.9097 339.0354 339.1455 339.2669 339.3389
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.min]) # CV-RMSE minimum
```

```
## [1] 336.0759
```

```r
sqrt(fit_lasso_cv$cvm[fit_lasso_cv$lambda == fit_lasso_cv$lambda.1se]) # CV-RMSE one SE
```

```
## [1] 364.8996
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
## 1  255.2820965 202094.5  22571.04  224665.6 179523.49     0
## 2  232.6035386 193864.7  23173.90  217038.6 170690.83     1
## 3  211.9396813 185305.1  22436.29  207741.4 162868.85     2
## 4  193.1115442 178122.2  21644.71  199766.9 156477.46     2
## 5  175.9560468 171726.0  21126.72  192852.7 150599.25     3
## 6  160.3245966 165527.2  20745.24  186272.4 144781.95     4
## 7  146.0818013 158605.1  20078.38  178683.5 138526.70     4
## 8  133.1042967 152647.9  19531.09  172179.0 133116.81     4
## 9  121.2796778 147734.8  19214.63  166949.4 128520.18     4
## 10 110.5055255 143788.4  19018.20  162806.6 124770.19     4
## 11 100.6885192 140560.0  18934.78  159494.7 121625.18     5
## 12  91.7436287 137729.5  18969.26  156698.7 118760.21     5
## 13  83.5933775 135271.4  19122.80  154394.2 116148.65     5
## 14  76.1671723 133151.7  19306.93  152458.7 113844.80     5
## 15  69.4006906 130964.1  19489.37  150453.5 111474.73     6
## 16  63.2353245 128616.4  19658.92  148275.3 108957.48     6
## 17  57.6176726 126439.9  19803.40  146243.3 106636.52     6
## 18  52.4990774 124644.7  19966.53  144611.2 104678.15     6
## 19  47.8352040 123169.3  20139.86  143309.1 103029.43     6
## 20  43.5856563 121958.1  20317.74  142275.8 101640.33     6
## 21  39.7136268 120958.1  20496.94  141455.1 100461.17     6
## 22  36.1855776 120138.4  20677.40  140815.8  99461.00     6
## 23  32.9709506 119538.2  20872.57  140410.7  98665.59     6
## 24  30.0419022 119097.7  21072.25  140169.9  98025.43     6
## 25  27.3730624 118745.4  21260.44  140005.8  97484.95     6
## 26  24.9413150 118459.9  21436.21  139896.1  97023.71     6
## 27  22.7255973 118252.6  21592.40  139845.0  96660.20     6
## 28  20.7067179 118092.5  21739.18  139831.6  96353.28     6
## 29  18.8671902 117977.9  21874.85  139852.7  96103.01     6
## 30  17.1910810 117869.8  22004.58  139874.4  95865.26     7
## 31  15.6638727 117796.4  22127.17  139923.6  95669.25     7
## 32  14.2723374 117779.1  22250.89  140030.0  95528.23     7
## 33  13.0044223 117782.6  22380.79  140163.3  95401.77     9
## 34  11.8491453 117798.1  22496.48  140294.6  95301.62     9
## 35  10.7964999 117886.2  22576.40  140462.6  95309.78     9
## 36   9.8373686 118130.6  22615.45  140746.1  95515.18     9
## 37   8.9634439 118449.9  22643.81  141093.8  95806.14     9
## 38   8.1671562 118725.1  22684.38  141409.5  96040.76    11
## 39   7.4416086 118651.8  22698.65  141350.5  95953.18    11
## 40   6.7805166 118236.8  22664.97  140901.7  95571.80    12
## 41   6.1781542 117745.4  22593.82  140339.2  95151.57    12
## 42   5.6293040 117074.8  22497.40  139572.2  94577.38    13
## 43   5.1292121 116249.4  22288.54  138537.9  93960.86    13
## 44   4.6735471 115444.0  22019.11  137463.1  93424.93    13
## 45   4.2583620 114805.5  21776.88  136582.4  93028.60    13
## 46   3.8800609 114274.8  21560.41  135835.2  92714.41    13
## 47   3.5353670 113812.3  21370.27  135182.5  92441.98    13
## 48   3.2212947 113455.4  21201.62  134657.0  92253.74    13
## 49   2.9351238 113191.5  21067.25  134258.8  92124.25    13
## 50   2.6743755 113039.1  20958.50  133997.6  92080.63    13
## 51   2.4367913 112947.0  20858.28  133805.3  92088.76    13
## 52   2.2203135 112950.9  20775.74  133726.6  92175.15    14
## 53   2.0230670 113009.4  20709.17  133718.6  92300.26    15
## 54   1.8433433 113133.5  20663.34  133796.8  92470.14    15
## 55   1.6795857 113251.2  20641.57  133892.8  92609.65    17
## 56   1.5303760 113310.7  20629.48  133940.1  92681.18    17
## 57   1.3944216 113331.0  20635.66  133966.7  92695.38    17
## 58   1.2705450 113377.4  20637.08  134014.4  92740.28    17
## 59   1.1576733 113399.6  20600.58  134000.2  92799.00    17
## 60   1.0548288 113431.3  20550.42  133981.7  92880.84    17
## 61   0.9611207 113546.3  20507.83  134054.1  93038.44    17
## 62   0.8757374 113699.6  20467.46  134167.1  93232.17    17
## 63   0.7979393 113854.2  20432.48  134286.7  93421.71    17
## 64   0.7270526 114005.9  20402.20  134408.1  93603.68    17
## 65   0.6624632 114138.6  20377.63  134516.2  93760.98    18
## 66   0.6036118 114294.3  20351.85  134646.2  93942.47    18
## 67   0.5499886 114409.6  20334.89  134744.5  94074.67    18
## 68   0.5011291 114560.6  20311.74  134872.4  94248.90    17
## 69   0.4566102 114671.4  20305.07  134976.5  94366.31    18
## 70   0.4160462 114780.0  20269.48  135049.5  94510.56    18
## 71   0.3790858 114859.8  20262.62  135122.4  94597.18    18
## 72   0.3454089 114945.0  20224.60  135169.6  94720.40    18
## 73   0.3147237 115019.7  20226.36  135246.1  94793.34    18
## 74   0.2867645 115102.1  20201.36  135303.4  94900.70    18
## 75   0.2612891 115150.9  20193.23  135344.1  94957.67    18
```

```r
glance(fit_lasso_cv) # the two lambda values of interest
```

```
##   lambda.min lambda.1se
## 1   2.436791   76.16717
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
