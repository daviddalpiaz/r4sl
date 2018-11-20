# Regularized Discriminant Analysis

We now use the  `Sonar` dataset from the `mlbench` package to explore a new regularization method, **regularized discriminant analysis** (RDA), which combines the LDA and QDA. This is similar to how elastic net combines the ridge and lasso.

## Sonar Data


```r
# this is a temporary workaround for an issue with glmnet, Matrix, and R version 3.3.3
# see here: http://stackoverflow.com/questions/43282720/r-error-in-validobject-object-when-running-as-script-but-not-in-console
library(methods)
```


```r
library(mlbench)
library(caret)
library(glmnet)
library(klaR)
```


```r
data(Sonar)
```


```r
#View(Sonar)
```


```r
table(Sonar$Class) / nrow(Sonar)
```

```
## 
##         M         R 
## 0.5336538 0.4663462
```


```r
ncol(Sonar) - 1
```

```
## [1] 60
```

## RDA

Regularized discriminant analysis uses the same general setup as LDA and QDA but estimates the covariance in a new way, which combines the covariance of QDA $(\hat{\Sigma}_k)$ with the covariance of LDA $(\hat{\Sigma})$ using a tuning parameter $\lambda$.

$$
\hat{\Sigma}_k(\lambda) = (1-\lambda)\hat{\Sigma}_k + \lambda \hat{\Sigma}
$$

Using the `rda()` function from the `klaR` package, which `caret` utilizes, makes an additional modification to the covariance matrix, which also has a tuning parameter $\gamma$.

$$
\hat{\Sigma}_k(\lambda,\gamma) = (1 -\gamma) \hat{\Sigma}_k(\lambda) + \gamma \frac{1}{p} \text{tr}(\hat{\Sigma}_k(\lambda)) I
$$

Both $\gamma$ and $\lambda$ can be thought of as mixing parameters, as they both take values between 0 and 1. For the four extremes of $\gamma$ and $\lambda$, the covariance structure reduces to special cases:

- $(\gamma=0, \lambda=0)$: QDA - individual covariance for each group.
- $(\gamma=0, \lambda=1)$: LDA - a common covariance matrix.
- $(\gamma=1, \lambda=0)$: Conditional independent variables - similar to Naive Bayes, but variable variances within group (main diagonal elements) are all equal.
- $(\gamma=1, \lambda=1)$: Classification using euclidean distance - as in previous case, but variances are the same for all groups. Objects are assigned to group with nearest mean.


## RDA with Grid Search


```r
set.seed(1337)
cv_5_grid = trainControl(method = "cv", number = 5)
```


```r
set.seed(1337)
fit_rda_grid = train(Class ~ ., data = Sonar, method = "rda", trControl = cv_5_grid)
fit_rda_grid
```

```
## Regularized Discriminant Analysis 
## 
## 208 samples
##  60 predictor
##   2 classes: 'M', 'R' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 166, 166, 167, 166, 167 
## Resampling results across tuning parameters:
## 
##   gamma  lambda  Accuracy   Kappa    
##   0.0    0.0     0.7013937  0.3841061
##   0.0    0.5     0.7739837  0.5473212
##   0.0    1.0     0.7411150  0.4789646
##   0.5    0.0     0.8217189  0.6390489
##   0.5    0.5     0.8077816  0.6095537
##   0.5    1.0     0.7845528  0.5670147
##   1.0    0.0     0.6778165  0.3535033
##   1.0    0.5     0.6875726  0.3738076
##   1.0    1.0     0.6875726  0.3738076
## 
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were gamma = 0.5 and lambda = 0.
```


```r
plot(fit_rda_grid)
```

![](31-rda_files/figure-latex/unnamed-chunk-9-1.pdf)<!-- --> 

## RDA with Random Search Search



```r
set.seed(1337)
cv_5_rand = trainControl(method = "cv", number = 5, search = "random")
```


```r
fit_rda_rand = train(Class ~ ., data = Sonar, method = "rda",
                     trControl = cv_5_rand, tuneLength = 9)
fit_rda_rand
```

```
## Regularized Discriminant Analysis 
## 
## 208 samples
##  60 predictor
##   2 classes: 'M', 'R' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 166, 166, 167, 166, 167 
## Resampling results across tuning parameters:
## 
##   gamma       lambda      Accuracy   Kappa    
##   0.07399023  0.99371759  0.7796748  0.5556869
##   0.14604362  0.33913968  0.8362369  0.6705529
##   0.24540405  0.92379666  0.8133566  0.6231035
##   0.28111731  0.97238848  0.7989547  0.5939312
##   0.33131745  0.98132543  0.7941928  0.5848112
##   0.37327926  0.19398230  0.8169570  0.6298688
##   0.45386562  0.82735873  0.8178862  0.6318771
##   0.56474213  0.97943029  0.7940767  0.5857574
##   0.94763002  0.02522857  0.7740999  0.5435202
## 
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were gamma = 0.1460436 and lambda
##  = 0.3391397.
```


```r
ggplot(fit_rda_rand)
```

![](31-rda_files/figure-latex/unnamed-chunk-12-1.pdf)<!-- --> 


## Comparison to Elastic Net


```r
set.seed(1337)
fit_elnet_grid = train(Class ~ ., data = Sonar, method = "glmnet",
                       trControl = cv_5_grid, tuneLength = 10)
```


```r
set.seed(1337)
fit_elnet_int_grid = train(Class ~ . ^ 2, data = Sonar, method = "glmnet",
                           trControl = cv_5_grid, tuneLength = 10)
```


## Results


```r
get_best_result = function(caret_fit) {
  best_result = caret_fit$results[as.numeric(rownames(caret_fit$bestTune)), ]
  rownames(best_result) = NULL
  best_result
}
```


```r
knitr::kable(rbind(
  get_best_result(fit_rda_grid),
  get_best_result(fit_rda_rand)))
```


\begin{tabular}{r|r|r|r|r|r}
\hline
gamma & lambda & Accuracy & Kappa & AccuracySD & KappaSD\\
\hline
0.5000000 & 0.0000000 & 0.8217189 & 0.6390489 & 0.0455856 & 0.0926920\\
\hline
0.1460436 & 0.3391397 & 0.8362369 & 0.6705529 & 0.0631932 & 0.1255389\\
\hline
\end{tabular}


```r
knitr::kable(rbind(
  get_best_result(fit_elnet_grid),
  get_best_result(fit_elnet_int_grid)))
```


\begin{tabular}{r|r|r|r|r|r}
\hline
alpha & lambda & Accuracy & Kappa & AccuracySD & KappaSD\\
\hline
0.4 & 0.0065641 & 0.8034843 & 0.6041866 & 0.0645470 & 0.1297952\\
\hline
0.1 & 0.0243225 & 0.8418118 & 0.6809599 & 0.0539204 & 0.1088486\\
\hline
\end{tabular}


## External Links

- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) - Paper justifying random tuning parameter search.
- [Random Hyperparameter Search](https://topepo.github.io/caret/random-hyperparameter-search.html) - Details on random tuning parameter search in `caret`.


## RMarkdown

The RMarkdown file for this chapter can be found [**here**](31-rda.Rmd). The file was created using `R` version 3.5.1 and the following packages:

- Base Packages, Attached


```
## [1] "stats"     "graphics"  "grDevices" "utils"     "datasets"  "methods"  
## [7] "base"
```

- Additional Packages, Attached


```
## [1] "klaR"    "MASS"    "glmnet"  "foreach" "Matrix"  "caret"   "ggplot2"
## [8] "lattice" "mlbench"
```

- Additional Packages, Not Attached


```
##  [1] "Rcpp"         "lubridate"    "class"        "assertthat"  
##  [5] "rprojroot"    "digest"       "ipred"        "mime"        
##  [9] "R6"           "plyr"         "backports"    "stats4"      
## [13] "e1071"        "evaluate"     "highr"        "pillar"      
## [17] "rlang"        "lazyeval"     "rstudioapi"   "data.table"  
## [21] "miniUI"       "rpart"        "combinat"     "rmarkdown"   
## [25] "labeling"     "splines"      "gower"        "stringr"     
## [29] "questionr"    "munsell"      "shiny"        "compiler"    
## [33] "httpuv"       "xfun"         "pkgconfig"    "htmltools"   
## [37] "nnet"         "tidyselect"   "tibble"       "prodlim"     
## [41] "bookdown"     "codetools"    "later"        "crayon"      
## [45] "dplyr"        "withr"        "recipes"      "ModelMetrics"
## [49] "grid"         "xtable"       "nlme"         "gtable"      
## [53] "magrittr"     "scales"       "stringi"      "reshape2"    
## [57] "promises"     "bindrcpp"     "timeDate"     "generics"    
## [61] "lava"         "iterators"    "tools"        "glue"        
## [65] "purrr"        "survival"     "yaml"         "colorspace"  
## [69] "knitr"        "bindr"
```

