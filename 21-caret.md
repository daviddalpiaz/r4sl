# The `caret` Package



**Instructor's Note: This chapter is currently missing the usual narrative text. Hopefully it will be added later.**

Now that we have seen a number of classification (and regression) methods, and introduced cross-validation, we see the general outline of a predictive analysis:

- Select a method
- Test-train split the available data
- Decide on a set of candidate models via tuning parameters
- Select the "best model" (chose the values of the tuning parameters) using a resampled metric with the training data
- Use chosen model to make predictions
- Calculate relevant metrics on the test data

At face value it would seem like it should be easy to repeat this process for a number of different methods, however we have run into a number of difficulties attempting to do so with `R`.

- The `predict()` function seems to have a different behavior for each new method we see.
- Many methods have different cross-validation functions, or worse yet, no built-in process for cross-validation.
- Not all methods expect the same data format. Some methods do not use formula syntax. 
- Different methods have different handling of categorical predictors. Some methods cannot handle factor variables.

Thankfully, the `R` community has essentially provided a silver bullet for these issues, the [`caret`](http://topepo.github.io/caret/) package. Returning to the above list, we will see that a number of these tasks are directly addressed in the `caret` package.

- Test-train split the available data
    - `createDataPartition()` will take the place of our manual data splitting. It will also do some extra work to ensure that the train and test samples are somewhat similar.
- Decide on a set of candidate models via tuning parameters
    - `expand.grid()` is not a function in `caret`, but we will get in the habit of using it to specify a grid of tuning parameters.
- Select the best model (tuning parameters) using a resampled metric
    - `trainControl()` will setup cross-validation
    - `train()` is the workhorse of `caret`. It takes the following information then trains the requested model:
        - `form`, a formula, such as `y ~ .`
        - `data`, the data used for training
        - `method`, a statistical learning method from [a long list of availible models](https://topepo.github.io/caret/available-models.html)
        - `preProcess` which allows for specification of data pre-processing such as centering and scaling
        - `tuneGrid` which specifies the tuning parameters to train over
        - `trControl` which specifies the resampling scheme, that is, how cross-validation should be performed to find the best values of the tuning parameters
- Use chosen model to make predictions
    - `predict()` used on objects of type `train` will be truly magical!

## Classification Example

To illustrate `caret`, we will use the `Default` data from the `ISLR` package.


```r
data(Default, package = "ISLR")
```


```r
library(caret)
```

We first test-train split the data using `createDataPartition`. Here we are using 75% of the data for training.


```r
set.seed(430)
default_idx = createDataPartition(Default$default, p = 0.75, list = FALSE)
default_trn = Default[default_idx, ]
default_tst = Default[-default_idx, ]
```

- TODO: Why are we specifying the response? (`createDataPartition` tries to create similiar distributions in the train and test sets.)


```r
default_glm = train(
  form = default ~ .,
  data = default_trn,
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 5)
)
```


```r
default_glm
```

```
## Generalized Linear Model 
## 
## 7501 samples
##    3 predictor
##    2 classes: 'No', 'Yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 6001, 6001, 6000, 6001, 6001 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9733372  0.4174282
```


```r
names(default_glm)
```

```
##  [1] "method"       "modelInfo"    "modelType"    "results"     
##  [5] "pred"         "bestTune"     "call"         "dots"        
##  [9] "metric"       "control"      "finalModel"   "preProcess"  
## [13] "trainingData" "resample"     "resampledCM"  "perfNames"   
## [17] "maximize"     "yLimits"      "times"        "levels"      
## [21] "terms"        "coefnames"    "contrasts"    "xlevels"
```


```r
default_glm$results
```

```
##   parameter  Accuracy     Kappa AccuracySD   KappaSD
## 1      none 0.9733372 0.4174282 0.00358649 0.1180854
```


```r
default_glm$finalModel
```

```
## 
## Call:  NULL
## 
## Coefficients:
## (Intercept)   studentYes      balance       income  
##  -1.066e+01   -6.254e-01    5.647e-03    1.395e-06  
## 
## Degrees of Freedom: 7500 Total (i.e. Null);  7497 Residual
## Null Deviance:	    2192 
## Residual Deviance: 1204 	AIC: 1212
```


```r
calc_acc = function(actual, predicted) {
  mean(actual == predicted)
}
```


```r
# make predictions
head(predict(default_glm, newdata = default_trn))
```

```
## [1] No No No No No No
## Levels: No Yes
```


```r
# train acc
calc_acc(actual = default_trn$default,
         predicted = predict(default_glm, newdata = default_trn))
```

```
## [1] 0.9730703
```


```r
# test acc
calc_acc(actual = default_tst$default,
         predicted = predict(default_glm, newdata = default_tst))
```

```
## [1] 0.9739896
```


```r
# get probs
head(predict(default_glm, newdata = default_trn, type = "prob"))
```

```
##          No         Yes
## 1 0.9984674 0.001532637
## 3 0.9895850 0.010414985
## 5 0.9979141 0.002085863
## 6 0.9977233 0.002276746
## 8 0.9987645 0.001235527
## 9 0.9829081 0.017091877
```

- TODO: WOW


```r
default_knn = train(
  default ~ .,
  data = default_trn,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5)
)
```


```r
default_knn
```

```
## k-Nearest Neighbors 
## 
## 7501 samples
##    3 predictor
##    2 classes: 'No', 'Yes' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 6001, 6000, 6001, 6001, 6001 
## Resampling results across tuning parameters:
## 
##   k  Accuracy   Kappa     
##   5  0.9660044  0.14910366
##   7  0.9654711  0.08890944
##   9  0.9660044  0.03400684
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 9.
```


```r
default_knn = train(
  default ~ .,
  data = default_trn,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5),
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(k = seq(1, 101, by = 2))
)
```


```r
# default_knn
```


```r
plot(default_knn)
```



\begin{center}\includegraphics{21-caret_files/figure-latex/unnamed-chunk-18-1} \end{center}


```r
ggplot(default_knn) + theme_bw()
```



\begin{center}\includegraphics{21-caret_files/figure-latex/unnamed-chunk-19-1} \end{center}


```r
default_knn$bestTune
```

```
##    k
## 9 17
```


```r
# only works for single parameter
get_best_result = function(caret_fit) {
  best_result = caret_fit$results[as.numeric(rownames(caret_fit$bestTune)), ]
  rownames(best_result) = NULL
  best_result
}
```


```r
get_best_result(default_knn)
```

```
##    k  Accuracy     Kappa  AccuracySD    KappaSD
## 1 17 0.9720036 0.3803205 0.002404977 0.05972573
```


```r
default_knn$finalModel
```

```
## 17-nearest neighbor classification model
## Training set class distribution:
## 
##   No  Yes 
## 7251  250
```



## Regression Example


```r
gen_some_data = function(n_obs = 50) {
  x1 = seq(0, 10, length.out = n_obs)
  x2 = runif(n = n_obs, min = 0, max = 2)
  x3 = sample(c("A", "B", "C"), size = n_obs, replace = TRUE)
  x4 = round(runif(n = n_obs, min = 0, max = 5), 1)
  x5 = round(runif(n = n_obs, min = 0, max = 5), 0)
  y = round(x1 ^ 2 + x2 ^ 2 + 2 * (x3 == "B") + rnorm(n = n_obs), 3)
  data.frame(y, x1, x2, x3, x4, x5)
}
```


```r
set.seed(42)
sim_trn = gen_some_data(n_obs = 500)
sim_tst = gen_some_data(n_obs = 5000)
```


```r
sim_knn = train(
  y ~ .,
  data = sim_trn,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5),
  # preProcess = c("center", "scale"),
  tuneGrid = expand.grid(k = seq(1, 31, by = 2))
)
```

- TODO: Why no scaling?


```r
plot(sim_knn)
```



\begin{center}\includegraphics{21-caret_files/figure-latex/unnamed-chunk-27-1} \end{center}




\begin{center}\includegraphics{21-caret_files/figure-latex/unnamed-chunk-28-1} \end{center}

- TODO: 1se rule
- TODO: is this any good?

### New Methods


```r
gbm_grid = expand.grid(interaction.depth = c(1, 2, 3),
                       n.trees = (1:30) * 100,
                       shrinkage = c(0.1, 0.3),
                       n.minobsinnode = 20)
head(gbm_grid)
```

```
##   interaction.depth n.trees shrinkage n.minobsinnode
## 1                 1     100       0.1             20
## 2                 2     100       0.1             20
## 3                 3     100       0.1             20
## 4                 1     200       0.1             20
## 5                 2     200       0.1             20
## 6                 3     200       0.1             20
```


```r
set.seed(42)
sim_gbm_mod = train(
  y ~ .,
  data = sim_trn,
  method = "gbm",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = gbm_grid, 
  verbose = FALSE
)
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.3
```

```r
plot(sim_gbm_mod)
```



\begin{center}\includegraphics{21-caret_files/figure-latex/unnamed-chunk-30-1} \end{center}


```r
sim_gbm_mod$bestTune
```

```
##    n.trees interaction.depth shrinkage n.minobsinnode
## 30    3000                 1       0.1             20
```


```r
min(sim_gbm_mod$results$RMSE)
```

```
## [1] 1.912962
```


```r
calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
```


```r
calc_rmse(actual = sim_tst$y,
          predicted = predict(sim_gbm_mod, sim_tst))
```

```
## [1] 1.513519
```



- TODO: issues with this tuning?
- TODO: did we tune enough?
- TODO: is this any good?
- TODO: how are factors being used?




```r
sim_lm_mod = train(
  y ~ x1 + I(x1^2) + x2 + I(x2^2) + x3,
  data = sim_trn,
  method = "lm",
  trControl = trainControl(method = "cv", number = 5)
)
```




```r
sim_lm_mod$results$RMSE
```

```
## [1] 0.9806239
```


```r
calc_rmse(actual = sim_tst$y,
          predicted = predict(sim_lm_mod, sim_tst))
```

```
## [1] 1.014825
```

Notes to add later:

- Default grid vs specified grid. `tuneLength`
- Create table summarizing results for `knn()` and `glm()`. Test, train, and CV accuracy. Maybe also show SD for CV.

## External Links

- [The `caret` Package](http://topepo.github.io/caret/index.html) - Reference documentation for the `caret` package in `bookdown` format.
- [`caret` Model List](http://topepo.github.io/caret/available-models.html) - List of available models in `caret`.


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](21-caret.Rmd). The file was created using `R` version 3.4.2.
