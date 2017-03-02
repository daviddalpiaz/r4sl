# The `caret` Package

**Instructor's Note: This chapter is currently missing the usual narrative text. Hopefully it will be added later.**

Now that we have seen a number of classification (and regression) methods, and introduced cross-validation, we see the general outline of a predictive analysis:

- Select a method
- Test-train split the available data
- Decide on a set of candidate models via tuning parameters
- Select the best model (tuning parameters) using a cross-validated metric
- Use chosen model to make predictions
- Calculate relevant metrics on the test data

At face value it would seem like it should be easy to repeat this process for a number of different methods, however we have run into a number of difficulties attempting to do so with `R`.

- The `predict()` function seems to have a different behavior for each new method we see.
- Many methods have different cross-validation functions, or worse yet, no built-in process for cross-validation.
- Not all methods expect the same data format. Some methods do not use formula syntax.
- Different methods have different handling of categorical predictors.

Thankfully, the `R` community has essentially provided a silver bullet for these issues, the [`caret`](http://topepo.github.io/caret/) package. Returning to the above list, we will see that a number of these tasks are directly addressed in the `caret` package.

- Test-train split the available data
    - `createDataPartition()` will take the place of our manual data splitting. It will also do some extra work to ensure that the train and test samples are somewhat similar.
- Decide on a set of candidate models via tuning parameters
    - `expand.grid()` is not a function in `caret`, but we will get in the habit of using it to specify a grid of tuning parameters.
- Select the best model (tuning parameters) using a cross-validated metric
    - `trainControl()` will setup cross-validation
    - `train()` is the workhorse of `caret`. It takes the following information then trains the requested model:
        - `form`, a formula, such as `y ~ .`
        - `data`
        - `method`, from a long list of possibilities
        - `preProcess` which allows for specification of things such as centering and scaling
        - `tuneGrid` which specifies the tuning parameters to train over
        - `trControl` which specifies the resampling scheme, that is, how cross-validation should be performed
- Use chosen model to make predictions
    - `predict()` used on objects of type `train` will be magical!

To illustrate `caret`, we return to our familiar `Default` data.


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
## Summary of sample sizes: 6000, 6001, 6001, 6001, 6001 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9729372  0.4147209
## 
## 
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
##   parameter  Accuracy     Kappa  AccuracySD    KappaSD
## 1      none 0.9729372 0.4147209 0.001527574 0.04620646
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
accuracy = function(actual, predicted) {
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
accuracy(actual = default_trn$default,
         predicted = predict(default_glm, newdata = default_trn))
```

```
## [1] 0.9730703
```


```r
# test acc
accuracy(actual = default_tst$default,
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
## Summary of sample sizes: 6000, 6001, 6001, 6001, 6001 
## Resampling results across tuning parameters:
## 
##   k  Accuracy   Kappa    
##   5  0.9656046  0.1770996
##   7  0.9657378  0.1295425
##   9  0.9676045  0.1092291
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
  tuneGrid = expand.grid(k = seq(1, 100, by = 1))
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
## Pre-processing: centered (3), scaled (3) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 6001, 6001, 6001, 6001, 6000 
## Resampling results across tuning parameters:
## 
##   k    Accuracy   Kappa        
##     1  0.9566725   0.3102526809
##     2  0.9557393   0.3105592402
##     3  0.9676047   0.3737110726
##     4  0.9688045   0.3905217578
##     5  0.9693378   0.3720650032
##     6  0.9706708   0.3878199028
##     7  0.9705374   0.3805478527
##     8  0.9708038   0.3847869416
##     9  0.9718704   0.4008912524
##    10  0.9716037   0.3887596806
##    11  0.9718702   0.3893164746
##    12  0.9720036   0.4017431900
##    13  0.9722702   0.3885361750
##    14  0.9722703   0.3944481266
##    15  0.9730702   0.4007241418
##    16  0.9720039   0.3811342289
##    17  0.9722703   0.3792775497
##    18  0.9720038   0.3788523329
##    19  0.9720038   0.3697985519
##    20  0.9721370   0.3740347410
##    21  0.9718704   0.3615639366
##    22  0.9718705   0.3680145319
##    23  0.9717371   0.3539511129
##    24  0.9717371   0.3541232524
##    25  0.9717371   0.3499644208
##    26  0.9716037   0.3455285320
##    27  0.9720037   0.3528854219
##    28  0.9722703   0.3638828006
##    29  0.9724036   0.3687985599
##    30  0.9726703   0.3779447479
##    31  0.9722704   0.3584335333
##    32  0.9721371   0.3547676501
##    33  0.9718704   0.3473184965
##    34  0.9713372   0.3291867049
##    35  0.9718703   0.3367666962
##    36  0.9716038   0.3260879521
##    37  0.9717371   0.3320811820
##    38  0.9716037   0.3215128939
##    39  0.9716037   0.3219513284
##    40  0.9717371   0.3276712249
##    41  0.9718704   0.3277674417
##    42  0.9717371   0.3188637072
##    43  0.9714704   0.3066997114
##    44  0.9714704   0.3077206051
##    45  0.9710704   0.2961400483
##    46  0.9712037   0.2961184626
##    47  0.9710704   0.2914235196
##    48  0.9709372   0.2860070606
##    49  0.9708037   0.2847306864
##    50  0.9708037   0.2847306864
##    51  0.9708036   0.2812029517
##    52  0.9710703   0.2867649638
##    53  0.9708037   0.2771131971
##    54  0.9709372   0.2817510947
##    55  0.9709371   0.2781721329
##    56  0.9709371   0.2774089321
##    57  0.9706705   0.2714234284
##    58  0.9706704   0.2638759429
##    59  0.9706703   0.2638759634
##    60  0.9706703   0.2585241342
##    61  0.9701371   0.2368125620
##    62  0.9701372   0.2327890464
##    63  0.9701372   0.2327890464
##    64  0.9701372   0.2279714026
##    65  0.9697372   0.2110842159
##    66  0.9694708   0.1945370488
##    67  0.9697372   0.1951671666
##    68  0.9700038   0.2030358192
##    69  0.9696040   0.1849170701
##    70  0.9693373   0.1717143623
##    71  0.9690708   0.1588168415
##    72  0.9690708   0.1593832545
##    73  0.9688041   0.1399844247
##    74  0.9684042   0.1264961122
##    75  0.9684041   0.1201537186
##    76  0.9682708   0.1133391796
##    77  0.9681375   0.1068555860
##    78  0.9677375   0.0857547640
##    79  0.9677376   0.0921848301
##    80  0.9677376   0.0853775175
##    81  0.9680043   0.0866367370
##    82  0.9677377   0.0790611308
##    83  0.9677376   0.0725173945
##    84  0.9676043   0.0658085644
##    85  0.9676043   0.0651989918
##    86  0.9674709   0.0578805892
##    87  0.9676043   0.0651989918
##    88  0.9674709   0.0584901617
##    89  0.9676043   0.0651989918
##    90  0.9670710   0.0362615841
##    91  0.9669378   0.0277870004
##    92  0.9669377   0.0286699611
##    93  0.9668044   0.0210781703
##    94  0.9666711   0.0141321593
##    95  0.9666711   0.0073298429
##    96  0.9666711   0.0073298429
##    97  0.9665378  -0.0002617801
##    98  0.9665378  -0.0002617801
##    99  0.9666711   0.0000000000
##   100  0.9666711   0.0000000000
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was k = 15.
```


```r
plot(default_knn)
```

![](13-caret_files/figure-latex/unnamed-chunk-18-1.pdf)<!-- --> 


```r
ggplot(default_knn) + theme_bw()
```

![](13-caret_files/figure-latex/unnamed-chunk-19-1.pdf)<!-- --> 


```r
default_knn$bestTune
```

```
##     k
## 15 15
```


```r
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
## 1 15 0.9730702 0.4007241 0.002241574 0.08402323
```


```r
default_knn$finalModel
```

```
## 15-nearest neighbor classification model
## Training set class distribution:
## 
##   No  Yes 
## 7251  250
```

Notes to add later:

- Fewer ties with CV than simple test-train approach
- Default grid vs specified grid. `tuneLength`
- Create table summarizing results for `knn()` and `glm()`. Test, train, and CV accuracy. Maybe also show SD for CV.


## External Links

- [The `caret` Package](http://topepo.github.io/caret/index.html) - Reference documentation for the `caret` package in `bookdown` format.
- [`caret` Model List](http://topepo.github.io/caret/available-models.html) - List of available models in `caret`.


## RMarkdown

The RMarkdown file for this chapter can be found [**here**](13-caret.Rmd.Rmd). The file was created using `R` version 3.3.2 and the following packages:

- Base Packages, Attached


```
## [1] "methods"   "stats"     "graphics"  "grDevices" "utils"     "datasets" 
## [7] "base"
```

- Additional Packages, Attached


```
## [1] "caret"   "ggplot2" "lattice"
```

- Additional Packages, Not Attached


```
##  [1] "Rcpp"         "compiler"     "nloptr"       "plyr"        
##  [5] "class"        "iterators"    "tools"        "digest"      
##  [9] "lme4"         "evaluate"     "tibble"       "gtable"      
## [13] "nlme"         "mgcv"         "Matrix"       "foreach"     
## [17] "yaml"         "parallel"     "SparseM"      "e1071"       
## [21] "stringr"      "knitr"        "MatrixModels" "stats4"      
## [25] "rprojroot"    "grid"         "nnet"         "rmarkdown"   
## [29] "bookdown"     "minqa"        "reshape2"     "car"         
## [33] "magrittr"     "backports"    "scales"       "codetools"   
## [37] "ModelMetrics" "htmltools"    "MASS"         "splines"     
## [41] "assertthat"   "pbkrtest"     "colorspace"   "labeling"    
## [45] "quantreg"     "stringi"      "lazyeval"     "munsell"
```




