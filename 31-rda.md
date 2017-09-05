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
## Summary of sample sizes: 166, 167, 166, 167, 166 
## Resampling results across tuning parameters:
## 
##   gamma  lambda  Accuracy   Kappa    
##   0.0    0.0     0.7261324  0.4397685
##   0.0    0.5     0.7648084  0.5279282
##   0.0    1.0     0.7406504  0.4796821
##   0.5    0.0     0.7842044  0.5641761
##   0.5    0.5     0.8130081  0.6226443
##   0.5    1.0     0.7649245  0.5284504
##   1.0    0.0     0.6873403  0.3728292
##   1.0    0.5     0.6922184  0.3830140
##   1.0    1.0     0.6922184  0.3829488
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were gamma = 0.5 and lambda = 0.5.
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
## Summary of sample sizes: 166, 167, 166, 167, 166 
## Resampling results across tuning parameters:
## 
##   gamma      lambda     Accuracy   Kappa    
##   0.2091218  0.9853343  0.7986063  0.5944959
##   0.2306276  0.8632831  0.8177700  0.6328588
##   0.3223120  0.3194769  0.8275261  0.6509822
##   0.5074480  0.8843909  0.7842044  0.5654024
##   0.5274011  0.4747535  0.8178862  0.6323459
##   0.6146998  0.6471883  0.7937282  0.5828269
##   0.7031213  0.1969985  0.8178862  0.6304034
##   0.7363932  0.2499440  0.8177700  0.6295909
##   0.9860836  0.2297174  0.7207898  0.4384828
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were gamma = 0.322312 and lambda
##  = 0.3194769.
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
0.500000 & 0.5000000 & 0.8130081 & 0.6226443 & 0.0553439 & 0.1099849\\
\hline
0.322312 & 0.3194769 & 0.8275261 & 0.6509822 & 0.0650432 & 0.1322546\\
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
1.0 & 0.0350306 & 0.7984901 & 0.5953995 & 0.0652593 & 0.1311529\\
\hline
0.1 & 0.0243225 & 0.8321719 & 0.6617794 & 0.0744795 & 0.1480774\\
\hline
\end{tabular}


## External Links

- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) - Paper justifying random tuning parameter search.
- [Random Hyperparameter Search](https://topepo.github.io/caret/random-hyperparameter-search.html) - Details on random tuning parameter search in `caret`.


## RMarkdown

The RMarkdown file for this chapter can be found [**here**](17-rda.Rmd). The file was created using `R` version 3.4.1 and the following packages:

- Base Packages, Attached


```
## [1] "methods"   "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "base"
```

- Additional Packages, Attached


```
## [1] "klaR"    "MASS"    "glmnet"  "foreach" "Matrix"  "caret"   "ggplot2"
## [8] "lattice" "mlbench"
```

- Additional Packages, Not Attached


```
##  [1] "Rcpp"         "nloptr"       "compiler"     "plyr"        
##  [5] "class"        "iterators"    "tools"        "digest"      
##  [9] "lme4"         "evaluate"     "tibble"       "gtable"      
## [13] "nlme"         "mgcv"         "rlang"        "parallel"    
## [17] "yaml"         "SparseM"      "e1071"        "stringr"     
## [21] "knitr"        "MatrixModels" "combinat"     "stats4"      
## [25] "rprojroot"    "grid"         "nnet"         "rmarkdown"   
## [29] "bookdown"     "minqa"        "reshape2"     "car"         
## [33] "magrittr"     "backports"    "scales"       "codetools"   
## [37] "ModelMetrics" "htmltools"    "splines"      "pbkrtest"    
## [41] "colorspace"   "labeling"     "quantreg"     "stringi"     
## [45] "lazyeval"     "munsell"
```

