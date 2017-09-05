# Subset Selection

**Instructor's Note: This chapter is currently missing the usual narrative text. Hopefully it will be added later.**


```r
data(Hitters, package = "ISLR")
```


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

## AIC, BIC, and Cp

### `leaps` Package


```r
library(leaps)
```

### Best Subset


```r
fit_all = regsubsets(Salary ~ ., Hitters)
summary(fit_all)
```

```
## Subset selection object
## Call: regsubsets.formula(Salary ~ ., Hitters)
## 19 Variables  (and intercept)
##            Forced in Forced out
## AtBat          FALSE      FALSE
## Hits           FALSE      FALSE
## HmRun          FALSE      FALSE
## Runs           FALSE      FALSE
## RBI            FALSE      FALSE
## Walks          FALSE      FALSE
## Years          FALSE      FALSE
## CAtBat         FALSE      FALSE
## CHits          FALSE      FALSE
## CHmRun         FALSE      FALSE
## CRuns          FALSE      FALSE
## CRBI           FALSE      FALSE
## CWalks         FALSE      FALSE
## LeagueN        FALSE      FALSE
## DivisionW      FALSE      FALSE
## PutOuts        FALSE      FALSE
## Assists        FALSE      FALSE
## Errors         FALSE      FALSE
## NewLeagueN     FALSE      FALSE
## 1 subsets of each size up to 8
## Selection Algorithm: exhaustive
##          AtBat Hits HmRun Runs RBI Walks Years CAtBat CHits CHmRun CRuns
## 1  ( 1 ) " "   " "  " "   " "  " " " "   " "   " "    " "   " "    " "  
## 2  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
## 3  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
## 4  ( 1 ) " "   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
## 5  ( 1 ) "*"   "*"  " "   " "  " " " "   " "   " "    " "   " "    " "  
## 6  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   " "    " "   " "    " "  
## 7  ( 1 ) " "   "*"  " "   " "  " " "*"   " "   "*"    "*"   "*"    " "  
## 8  ( 1 ) "*"   "*"  " "   " "  " " "*"   " "   " "    " "   "*"    "*"  
##          CRBI CWalks LeagueN DivisionW PutOuts Assists Errors NewLeagueN
## 1  ( 1 ) "*"  " "    " "     " "       " "     " "     " "    " "       
## 2  ( 1 ) "*"  " "    " "     " "       " "     " "     " "    " "       
## 3  ( 1 ) "*"  " "    " "     " "       "*"     " "     " "    " "       
## 4  ( 1 ) "*"  " "    " "     "*"       "*"     " "     " "    " "       
## 5  ( 1 ) "*"  " "    " "     "*"       "*"     " "     " "    " "       
## 6  ( 1 ) "*"  " "    " "     "*"       "*"     " "     " "    " "       
## 7  ( 1 ) " "  " "    " "     "*"       "*"     " "     " "    " "       
## 8  ( 1 ) " "  "*"    " "     "*"       "*"     " "     " "    " "
```


```r
fit_all = regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
fit_all_sum = summary(fit_all)
names(fit_all_sum)
```

```
## [1] "which"  "rsq"    "rss"    "adjr2"  "cp"     "bic"    "outmat" "obj"
```


```r
fit_all_sum$bic
```

```
##  [1]  -90.84637 -128.92622 -135.62693 -141.80892 -144.07143 -147.91690
##  [7] -145.25594 -147.61525 -145.44316 -143.21651 -138.86077 -133.87283
## [13] -128.77759 -123.64420 -118.21832 -112.81768 -107.35339 -101.86391
## [19]  -96.30412
```


```r
par(mfrow = c(2, 2))
plot(fit_all_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "b")

plot(fit_all_sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "b")
best_adj_r2 = which.max(fit_all_sum$adjr2)
points(best_adj_r2, fit_all_sum$adjr2[best_adj_r2],
       col = "red",cex = 2, pch = 20)

plot(fit_all_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = 'b')
best_cp = which.min(fit_all_sum$cp)
points(best_cp, fit_all_sum$cp[best_cp], 
       col = "red", cex = 2, pch = 20)

plot(fit_all_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = 'b')
best_bic = which.min(fit_all_sum$bic)
points(best_bic, fit_all_sum$bic[best_bic], 
       col = "red", cex = 2, pch = 20)
```

![](22-subset_files/figure-latex/unnamed-chunk-7-1.pdf)<!-- --> 

### Stepwise Methods


```r
fit_fwd = regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
fit_fwd_sum = summary(fit_fwd)
```


```r
fit_bwd = regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "backward")
fit_bwd_sum = summary(fit_bwd)
```


```r
coef(fit_fwd, 7)
```

```
##  (Intercept)        AtBat         Hits        Walks         CRBI 
##  109.7873062   -1.9588851    7.4498772    4.9131401    0.8537622 
##       CWalks    DivisionW      PutOuts 
##   -0.3053070 -127.1223928    0.2533404
```

```r
coef(fit_bwd, 7)
```

```
##  (Intercept)        AtBat         Hits        Walks        CRuns 
##  105.6487488   -1.9762838    6.7574914    6.0558691    1.1293095 
##       CWalks    DivisionW      PutOuts 
##   -0.7163346 -116.1692169    0.3028847
```

```r
coef(fit_all, 7)
```

```
##  (Intercept)         Hits        Walks       CAtBat        CHits 
##   79.4509472    1.2833513    3.2274264   -0.3752350    1.4957073 
##       CHmRun    DivisionW      PutOuts 
##    1.4420538 -129.9866432    0.2366813
```


```r
fit_bwd_sum = summary(fit_bwd)
which.min(fit_bwd_sum$cp)
```

```
## [1] 10
```

```r
coef(fit_bwd, which.min(fit_bwd_sum$cp))
```

```
##  (Intercept)        AtBat         Hits        Walks       CAtBat 
##  162.5354420   -2.1686501    6.9180175    5.7732246   -0.1300798 
##        CRuns         CRBI       CWalks    DivisionW      PutOuts 
##    1.4082490    0.7743122   -0.8308264 -112.3800575    0.2973726 
##      Assists 
##    0.2831680
```


```r
fit = lm(Salary ~ ., data = Hitters)
fit_aic_back = step(fit, trace = FALSE)
coef(fit_aic_back)
```

```
##  (Intercept)        AtBat         Hits        Walks       CAtBat 
##  162.5354420   -2.1686501    6.9180175    5.7732246   -0.1300798 
##        CRuns         CRBI       CWalks    DivisionW      PutOuts 
##    1.4082490    0.7743122   -0.8308264 -112.3800575    0.2973726 
##      Assists 
##    0.2831680
```

## Validated RMSE


```r
set.seed(42)
num_vars = ncol(Hitters) - 1
trn_idx = sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)
tst_idx = (!trn_idx)

fit_all = regsubsets(Salary ~ ., data = Hitters[trn_idx, ], nvmax = num_vars)
test_mat = model.matrix(Salary ~ ., data = Hitters[tst_idx, ])

test_err = rep(0, times = num_vars)
for (i in seq_along(test_err)) {
  coefs = coef(fit_all, id = i)
  pred = test_mat[, names(coefs)] %*% coefs
  test_err[i] <- sqrt(mean((Hitters$Salary[tst_idx] - pred) ^ 2))
}
test_err
```

```
##  [1] 357.1226 333.8531 323.6408 320.5458 308.0303 295.1308 301.8142
##  [8] 309.2389 303.3976 307.9660 307.4841 306.9883 313.2374 314.3905
## [15] 313.8258 314.0586 313.6674 313.3490 313.3424
```


```r
plot(test_err, type='b', ylab = "Test Set RMSE", xlab = "Number of Predictors")
```

![](22-subset_files/figure-latex/unnamed-chunk-14-1.pdf)<!-- --> 


```r
which.min(test_err)
```

```
## [1] 6
```

```r
coef(fit_all, which.min(test_err))
```

```
##  (Intercept)        Walks       CAtBat        CHits         CRBI 
##  171.2082504    5.0067050   -0.4005457    1.2951923    0.7894534 
##    DivisionW      PutOuts 
## -131.1212694    0.2682166
```


```r
class(fit_all)
```

```
## [1] "regsubsets"
```


```r
predict.regsubsets = function(object, newdata, id, ...) {
  
  form  = as.formula(object$call[[2]])
  mat   = model.matrix(form, newdata)
  coefs = coef(object, id = id)
  xvars = names(coefs)
  
  mat[, xvars] %*% coefs
}
```


```r
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
```


```r
num_folds = 5
num_vars  = 19
set.seed(1)
folds = caret::createFolds(Hitters$Salary, k = num_folds)
fold_error = matrix(0, nrow = num_folds, ncol = num_vars, 
                    dimnames = list(paste(1:5), paste(1:19)))

for(j in 1:num_folds) {
  
  train_fold    = Hitters[-folds[[j]], ]
  validate_fold = Hitters[ folds[[j]], ]

  
  best_fit = regsubsets(Salary ~ ., data = train_fold, nvmax = 19)
  
  for (i in 1:num_vars) {
    
    pred = predict(best_fit, validate_fold, id = i)
    
    fold_error[j, i] = rmse(actual = validate_fold$Salary,
                            predicted = pred)
  }
  
}

cv_error = apply(fold_error, 2, mean)
cv_error
```

```
##        1        2        3        4        5        6        7        8 
## 373.2202 363.0715 374.8356 362.2405 357.6623 350.0238 348.0589 342.9089 
##        9       10       11       12       13       14       15       16 
## 343.8661 341.6405 339.4228 341.9303 342.5545 342.0155 340.8147 343.4722 
##       17       18       19 
## 343.4259 343.8129 343.2279
```


```r
plot(cv_error, type='b', ylab = "Corss-Validated RMSE", xlab = "Number of Predictors")
```

![](22-subset_files/figure-latex/unnamed-chunk-20-1.pdf)<!-- --> 


```r
fit_all = regsubsets(Salary ~ ., data = Hitters, nvmax = num_vars)
coef(fit_all, which.min(cv_error))
```

```
##  (Intercept)        AtBat         Hits        Walks       CAtBat 
##  135.7512195   -2.1277482    6.9236994    5.6202755   -0.1389914 
##        CRuns         CRBI       CWalks      LeagueN    DivisionW 
##    1.4553310    0.7852528   -0.8228559   43.1116152 -111.1460252 
##      PutOuts      Assists 
##    0.2894087    0.2688277
```


## External Links

- []() - 


## RMarkdown

The RMarkdown file for this chapter can be found [**here**](14-subset.Rmd). The file was created using `R` version 3.4.1 and the following packages:

- Base Packages, Attached


```
## [1] "stats"     "graphics"  "grDevices" "utils"     "datasets"  "base"
```

- Additional Packages, Attached


```
## [1] "leaps"
```

- Additional Packages, Not Attached


```
##  [1] "Rcpp"         "nloptr"       "compiler"     "plyr"        
##  [5] "methods"      "iterators"    "tools"        "digest"      
##  [9] "lme4"         "evaluate"     "tibble"       "gtable"      
## [13] "nlme"         "lattice"      "mgcv"         "rlang"       
## [17] "Matrix"       "foreach"      "parallel"     "yaml"        
## [21] "SparseM"      "stringr"      "knitr"        "MatrixModels"
## [25] "stats4"       "rprojroot"    "grid"         "caret"       
## [29] "nnet"         "rmarkdown"    "bookdown"     "minqa"       
## [33] "ggplot2"      "reshape2"     "car"          "magrittr"    
## [37] "backports"    "scales"       "codetools"    "ModelMetrics"
## [41] "htmltools"    "MASS"         "splines"      "pbkrtest"    
## [45] "colorspace"   "quantreg"     "stringi"      "lazyeval"    
## [49] "munsell"
```


