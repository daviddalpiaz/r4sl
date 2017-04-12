# Ensemble Methods

We'll now consider ensembles of trees. 

## Regression

We first consider the regression case, using the `Boston` data from the `MASS` package. We will use RMSE as our metric, so we write a function which will help us along the way.


```r
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
```

We also load all of the packages that we will need.


```r
library(tree)
library(MASS)
library(ISLR)
library(randomForest)
library(gbm)
library(caret)
```

We first test-train split the data and fit the same pruned tree as before. (Note: When pruning the tree, the best tree is actually the unpruned tree. View the results of `cv.tree` to see this. However, we select the tree of size 7 as the best of the pruned trees.)


```r
set.seed(18)
boston_idx = sample(1:nrow(Boston), nrow(Boston) / 2)
boston_trn = Boston[boston_idx,]
boston_tst = Boston[-boston_idx,]
```

### Tree Model


```r
boston_tree = tree(medv ~ ., data = boston_trn)
set.seed(18)
boston_tree_cv = cv.tree(boston_tree)
boston_tree_prune = prune.tree(boston_tree, best = 7)
boston_prune_tst_pred = predict(boston_tree_prune, newdata = boston_tst)
plot(boston_prune_tst_pred, boston_tst$medv, 
     xlab = "Predicted", ylab = "Actual", 
     main = "Predicted vs Actual: Tree, Test Data")
abline(0, 1, col = "red", lwd = 2)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-4-1.pdf)<!-- --> 

```r
(tree_tst_rmse = rmse(boston_prune_tst_pred, boston_tst$medv))
```

```
## [1] 5.331457
```

### Linear Model

Last time, we also fit an additive linear model, which we found to work better than the tree. The test RMSE is lower, and the predicted vs actual plot looks much better.


```r
bostom_lm = lm(medv ~ ., data = boston_trn)
boston_lm_tst_pred = predict(bostom_lm, newdata = boston_tst)
plot(boston_lm_tst_pred, boston_tst$medv,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Linear Model, Test Data"
)
abline(0, 1, col = "red", lwd = 2)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-5-1.pdf)<!-- --> 

```r
(lm_tst_rmse = rmse(boston_lm_tst_pred, boston_tst$medv))
```

```
## [1] 5.125877
```

### Bagging

We now fit a bagged model, using the `randomForest` package. Bagging is actually a special case of a random forest where `mtry` is equal to $p$, the number of predictors.


```r
boston_bag = randomForest(medv ~ ., data = boston_trn, mtry = 13, 
                          importance = TRUE, ntrees = 500)
boston_bag
```

```
## 
## Call:
##  randomForest(formula = medv ~ ., data = boston_trn, mtry = 13,      importance = TRUE, ntrees = 500) 
##                Type of random forest: regression
##                      Number of trees: 500
## No. of variables tried at each split: 13
## 
##           Mean of squared residuals: 14.20299
##                     % Var explained: 80.92
```

```r
boston_bag_tst_pred = predict(boston_bag, newdata = boston_tst)
plot(boston_bag_tst_pred,boston_tst$medv,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Bagged Model, Test Data"
)
abline(0, 1, col = "red", lwd = 2)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-6-1.pdf)<!-- --> 

```r
(bag_tst_rmse = rmse(boston_bag_tst_pred, boston_tst$medv))
```

```
## [1] 3.814368
```

Here we see two interesting results. First, the predicted versus actual plot no longer has a small number of predicted values. Second, our test error has dropped dramatically. Also note that the "Mean of squared residuals" which is output by `randomForest` is the **Out of Bag** estimate of the error.


```r
plot(boston_bag)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-7-1.pdf)<!-- --> 


### Random Forest

We now try a random forest. For regression, the suggestion is to use `mtry` equal to $p/3$.


```r
boston_forest = randomForest(medv ~ ., data = boston_trn, mtry = 4, 
                             importance = TRUE, ntrees = 500)
boston_forest
```

```
## 
## Call:
##  randomForest(formula = medv ~ ., data = boston_trn, mtry = 4,      importance = TRUE, ntrees = 500) 
##                Type of random forest: regression
##                      Number of trees: 500
## No. of variables tried at each split: 4
## 
##           Mean of squared residuals: 12.80737
##                     % Var explained: 82.79
```

```r
#importance(boston_forest)
#varImpPlot(boston_forest)
boston_forest_tst_pred = predict(boston_forest, newdata = boston_tst)
plot(boston_forest_tst_pred, boston_tst$medv,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Random Forest, Test Data"
)
abline(0, 1, col = "red", lwd = 2)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-8-1.pdf)<!-- --> 

```r
(forest_tst_rmse = rmse(boston_forest_tst_pred, boston_tst$medv))
```

```
## [1] 3.73447
```

```r
boston_forest_trn_pred = predict(boston_forest, newdata = boston_trn)
forest_trn_rmse = rmse(boston_forest_trn_pred, boston_trn$medv)
forest_oob_rmse = rmse(boston_forest$predicted, boston_trn$medv)
```

Here we note three RMSEs. The training RMSE (which is optimistic), the OOB RMSE (which is a good estimate of the test error) and the test RMSE. Also note that variables importance was calculated, however, the results are not shown here. (The code to view the results is commented out.)


```
##       Data    Error
## 1 Training 1.563253
## 2      OOB 3.578738
## 3     Test 3.734470
```


### Boosting

Lastly, we try a boosted model, which by default will produce a nice **variable importance** plot as well as plots of the marginal effects of the predictors. We use the `gbm` package.


```r
booston_boost = gbm(medv ~ ., data = boston_trn, distribution = "gaussian", 
                    n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
booston_boost
```

```
## gbm(formula = medv ~ ., distribution = "gaussian", data = boston_trn, 
##     n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
## A gradient boosted model with gaussian loss function.
## 5000 iterations were performed.
## There were 13 predictors of which 13 had non-zero influence.
```

```r
summary(booston_boost)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 

```
##             var    rel.inf
## rm           rm 33.1305117
## lstat     lstat 32.0413077
## dis         dis 10.1450348
## crim       crim  6.4535978
## black     black  4.7076459
## nox         nox  4.1647148
## age         age  3.4933055
## ptratio ptratio  2.1056374
## tax         tax  1.3320146
## indus     indus  0.9248479
## rad         rad  0.7800278
## chas       chas  0.5731764
## zn           zn  0.1481777
```

```r
par(mfrow = c(1, 2))
plot(booston_boost, i = "rm")
plot(booston_boost, i = "lstat")
```

![](20-ensemble_files/figure-latex/unnamed-chunk-10-2.pdf)<!-- --> 

```r
boston_boost_tst_pred = predict(booston_boost, newdata = boston_tst, n.trees = 5000)
(boost_tst_rmse = rmse(boston_boost_tst_pred, boston_tst$medv))
```

```
## [1] 3.437024
```


```r
plot(boston_boost_tst_pred, boston_tst$medv,
     xlab = "Predicted", ylab = "Actual", 
     main = "Predicted vs Actual: Boosted Model, Test Data"
)
abline(0, 1, col = "red", lwd = 2)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-11-1.pdf)<!-- --> 

### Results


```r
(boston_rmse = data.frame(
  Model = c("Single Tree", "Linear Model", "Bagging",  "Random Forest",  "Boosting"),
  TestError = c(tree_tst_rmse, lm_tst_rmse, bag_tst_rmse, forest_tst_rmse, boost_tst_rmse)
  )
)
```

```
##           Model TestError
## 1   Single Tree  5.331457
## 2  Linear Model  5.125877
## 3       Bagging  3.814368
## 4 Random Forest  3.734470
## 5      Boosting  3.437024
```

While a single tree does not beat linear regression, each of the ensemble methods perform much better!


## Classification

We now return to the `Carseats` dataset and the classification setting. We see that an additive logistic regression performs much better than a single tree, but we expect ensemble methods to bring trees closer to the logistic regression. Can they do better?

We now use prediction accuracy as our metric:


```r
accuracy = function(actual, predicted) {
  mean(actual == predicted)
}
```



```r
data(Carseats)
Carseats$Sales = as.factor(ifelse(Carseats$Sales <= 8, "Low", "High"))
set.seed(2)
seat_idx = sample(1:nrow(Carseats), 200)
seat_trn = Carseats[seat_idx,]
seat_tst = Carseats[-seat_idx,]
```

### Tree Model


```r
seat_tree = tree(Sales ~ ., data = seat_trn)
set.seed(3)
seat_tree_cv = cv.tree(seat_tree, FUN = prune.misclass)

seat_tree_prune = prune.misclass(seat_tree, best = 9)
seat_prune_tst_pred = predict(seat_tree_prune, seat_tst, type = "class")

table(predicted = seat_prune_tst_pred, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   60  22
##      Low    24  94
```

```r
(tree_tst_acc = accuracy(predicted = seat_prune_tst_pred, actual = seat_tst$Sales))
```

```
## [1] 0.77
```


### Logistic Regression



```r
seat_glm = glm(Sales ~ ., data = seat_trn, family = "binomial")
seat_glm_tst_pred = ifelse(predict(seat_glm, seat_tst, "response") > 0.5, 
                           "Low", "High")
table(predicted = seat_glm_tst_pred, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   75   9
##      Low     9 107
```

```r
(glm_tst_acc = accuracy(predicted = seat_glm_tst_pred, actual = seat_tst$Sales))
```

```
## [1] 0.91
```

### Bagging


```r
seat_bag = randomForest(Sales ~ ., data = seat_trn, mtry = 10, 
                        importance = TRUE, ntrees = 500)
seat_bag
```

```
## 
## Call:
##  randomForest(formula = Sales ~ ., data = seat_trn, mtry = 10,      importance = TRUE, ntrees = 500) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 10
## 
##         OOB estimate of  error rate: 21.5%
## Confusion matrix:
##      High Low class.error
## High   51  29   0.3625000
## Low    14 106   0.1166667
```

```r
seat_bag_tst_pred = predict(seat_bag, newdata = seat_tst)
table(predicted = seat_bag_tst_pred, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   68  21
##      Low    16  95
```

```r
(bag_tst_acc = accuracy(predicted = seat_bag_tst_pred, actual = seat_tst$Sales))
```

```
## [1] 0.815
```

### Random Forest

For classification, the suggested `mtry` for a random forest is $\sqrt{p}.$


```r
seat_forest = randomForest(Sales ~ ., data = seat_trn, mtry = 3, importance = TRUE, ntrees = 500)
seat_forest
```

```
## 
## Call:
##  randomForest(formula = Sales ~ ., data = seat_trn, mtry = 3,      importance = TRUE, ntrees = 500) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 3
## 
##         OOB estimate of  error rate: 22%
## Confusion matrix:
##      High Low class.error
## High   49  31   0.3875000
## Low    13 107   0.1083333
```

```r
seat_forest_tst_perd = predict(seat_forest, newdata = seat_tst)
table(predicted = seat_forest_tst_perd, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   63  16
##      Low    21 100
```

```r
(forest_tst_acc = accuracy(predicted = seat_forest_tst_perd, actual = seat_tst$Sales))
```

```
## [1] 0.815
```

### Boosting

To perform boosting, we modify the response to be `0` and `1` to work with `gbm`. Later we will use `caret` to fit `gbm` models, which will avoid this annoyance.


```r
seat_trn_mod = seat_trn
seat_trn_mod$Sales = as.numeric(ifelse(seat_trn_mod$Sales == "Low", "0", "1"))

seat_boost = gbm(Sales ~ ., data = seat_trn_mod, distribution = "bernoulli", 
                 n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
seat_boost
```

```
## gbm(formula = Sales ~ ., distribution = "bernoulli", data = seat_trn_mod, 
##     n.trees = 5000, interaction.depth = 4, shrinkage = 0.01)
## A gradient boosted model with bernoulli loss function.
## 5000 iterations were performed.
## There were 10 predictors of which 10 had non-zero influence.
```

```r
seat_boost_tst_pred = ifelse(predict(seat_boost, seat_tst, n.trees = 5000, "response") > 0.5, 
                             "High", "Low")
table(predicted = seat_boost_tst_pred, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   70  17
##      Low    14  99
```

```r
(boost_tst_acc = accuracy(predicted = seat_boost_tst_pred, actual = seat_tst$Sales))
```

```
## [1] 0.845
```


### Results


```r
(seat_acc = data.frame(
  Model = c("Single Tree", "Logistic Regression", "Bagging",  "Random Forest",  "Boosting"),
  TestAccuracy = c(tree_tst_acc, glm_tst_acc, bag_tst_acc, forest_tst_acc, boost_tst_acc)
  )
)
```

```
##                 Model TestAccuracy
## 1         Single Tree        0.770
## 2 Logistic Regression        0.910
## 3             Bagging        0.815
## 4       Random Forest        0.815
## 5            Boosting        0.845
```

Here we see each of the ensemble methods performing better than a single tree, however, they still fall behind logistic regression. Sometimes a simple linear model will beat more complicated models! This is why you should always try a logistic regression for classification.




## Tuning

So far we fit bagging, boosting and random forest models, but did not tune any of them, we simply used certain, somewhat arbitrary, parameters. Now we will see how to modify the tuning parameters to make these models better.

- Bagging: Actually just a subset of Random Forest with `mtry` = $p$.
- Random Forest: `mtry`
- Boosting: `n.trees`, `interaction.depth`, `shrinkage`, `n.minobsinnode`

We will use the `caret` package to accomplish this. Technically `ntrees` is a tuning parameter for both bagging and random forest, but `caret` will use 500 by default and there is no easy way to tune it. This will not make a big difference since for both we simply need "enough" and 500 seems to do the trick.

While `mtry` is a tuning parameter, there are suggested values for classification and regression:

- Regression: `mtry` = $p/3.$
- Classification: `mtry` = $\sqrt{p}.$

Also note that with these tree-based ensemble methods there are two resampling solutions for tuning the model:

- Out of Bag 
- Cross-Validation

Using Out of Bag samples is advantageous with these methods as compared to Cross-Validation since it removes the need to refit the model and is thus much more computationally efficient. Unfortunately OOB methods cannot be used with `gbm` models. See the [`caret` documentation](http://topepo.github.io/caret/training.html) for details.




### Random Forest and Bagging

Here we setup training control for both OOB and cross-validation methods. Note we specify `verbose = FALSE` which suppresses output related to progress. You may wish to set this to `TRUE` when first tuning a model since it will give you an idea of how long the tuning process will take. (Which can sometimes be a long time.)


```r
oob = trainControl(method = "oob")
cv_5 = trainControl(method = "cv", number = 5)
```

To tune a Random Forest in `caret` we will use `method = "rf"` which uses the `randomForest` function in the background. Here we elect to use the OOB training control that we created. We could also use Cross-Validation, however it will likely select a similar model, but requiring more time.

We setup a grid of `mtry` values which include all possible values since there are $10$ predictors in the dataset. An `mtry` of $10$ is actually bagging.


```r
dim(seat_trn)
```

```
## [1] 200  11
```

```r
rf_grid =  expand.grid(mtry = 1:10)
set.seed(825)
seat_rf_tune = train(Sales ~ ., data = seat_trn,
                     method = "rf",
                     trControl = oob,
                     verbose = FALSE,
                     tuneGrid = rf_grid)
seat_rf_tune
```

```
## Random Forest 
## 
## 200 samples
##  10 predictor
##   2 classes: 'High', 'Low' 
## 
## No pre-processing
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa    
##    1    0.750     0.4318182
##    2    0.785     0.5295405
##    3    0.785     0.5315904
##    4    0.805     0.5824411
##    5    0.785     0.5356371
##    6    0.810     0.5957447
##    7    0.810     0.5922747
##    8    0.805     0.5842217
##    9    0.785     0.5396146
##   10    0.795     0.5591398
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 6.
```

```r
accuracy(predict(seat_rf_tune, seat_tst), seat_tst$Sales)
```

```
## [1] 0.81
```

The results returned are based on the OOB samples. (Coincidentally, the test accuracy is the same as the best accuracy found using OOB samples.) Note that when using OOB, for some reason the default plot is not what you would expect and is not at all useful. (Which is why it is omitted here.)


```r
seat_rf_tune$bestTune
```

```
##   mtry
## 6    6
```

Based on these results, we would select the random forest model with an `mtry` of 6. Note that based on the OOB estimates, the bagging model is expected to perform worse than this select model, however, based on our results above, that is not what we find to be true in our test set.

Also note that `method = "ranger"` would also fit a random forest model. [Ranger](http://arxiv.org/pdf/1508.04409.pdf) is a newer `R` package for random forests that has been shown to be much faster, especially when there are a larger number of predictors.


### Boosting

We now tune a boosted tree model. We will use the cross-validation tune control setup above. We will fit the model using `gbm` with `caret`.

To setup the tuning grid, we must specify four parameters to tune:

- `interaction.depth`: How many splits to use with each tree.
- `n.trees`: The number of trees to use.
- `shrinkage`: The shrinkage parameters, which controls how fast the method learns.
- `n.minobsinnode`: The minimum number of observations in a node of the tree. (`caret` requires us to specify this. This is actually a tuning parameter of the trees, not boosting, and we would normally just accept the default.)

Finally, `expand.grid` comes in handy, as we can specify a vector of values for each parameter, then we get back a matrix of all possible combinations.


```r
gbm_grid =  expand.grid(interaction.depth = 1:5,
                        n.trees = (1:6) * 500,
                        shrinkage = c(0.001, 0.01, 0.1),
                        n.minobsinnode = 10)
```

We now train the model using all possible combinations of the tuning parameters we just specified.


```r
seat_gbm_tune = train(Sales ~ ., data = seat_trn,
                      method = "gbm",
                      trControl = cv_5,
                      verbose = FALSE,
                      tuneGrid = gbm_grid)
```

The additional `verbose = FALSE` in the `train` call suppresses additional output from each `gbm` call.

By default, calling `plot` here will produce a nice graphic of the results.


```r
#seat_gbm_tune
plot(seat_gbm_tune)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-26-1.pdf)<!-- --> 

```r
accuracy(predict(seat_gbm_tune, seat_tst), seat_tst$Sales)
```

```
## [1] 0.845
```

We see our tuned model does no better on the test set than the arbitrary boosted model we had fit above, with the slightly different parameters seen below. We could perhaps try a larger tuning grid, but at this point it seems unlikely that we could find a much better model. There seems to be no way to get a tree method to out-perform logistic regression in this dataset.


```r
seat_gbm_tune$bestTune
```

```
##    n.trees interaction.depth shrinkage n.minobsinnode
## 61     500                 1       0.1             10
```



## Tree versus Ensemble Boundaries


```r
library(mlbench)
set.seed(42)
sim_trn = mlbench.circle(n = 1000, d = 2)
sim_trn = data.frame(sim_trn$x, class = as.factor(sim_trn$classes))
sim_tst = mlbench.circle(n = 1000, d = 2)
sim_tst = data.frame(sim_tst$x, class = as.factor(sim_tst$classes))
```


```r
plot(sim_trn$X1, sim_trn$X2, col = sim_trn$class,
     xlab = "X1", ylab = "X2")
```

![](20-ensemble_files/figure-latex/unnamed-chunk-29-1.pdf)<!-- --> 


```r
cv_5 = trainControl(method = "cv", number = 5)
oob  = trainControl(method = "oob")
```


```r
sim_tree_cv = train(class ~ .,
                    data = sim_trn,
                    trControl = cv_5,
                    method = "rpart")
```

```
## Loading required package: rpart
```


```r
library(rpart.plot)
prp(sim_tree_cv$finalModel)
```

![](20-ensemble_files/figure-latex/unnamed-chunk-32-1.pdf)<!-- --> 


```r
rf_grid = expand.grid(mtry = c(1, 2))
sim_rf_oob = train(class ~ .,
                   data = sim_trn,
                   trControl = oob,
                   tuneGrid = rf_grid)
```


```r
gbm_grid =  expand.grid(interaction.depth = 1:5,
                        n.trees = (1:6) * 500,
                        shrinkage = c(0.001, 0.01, 0.1),
                        n.minobsinnode = 10)

sim_gbm_cv = train(class ~ ., 
                   data = sim_trn,
                   method = "gbm",
                   trControl = cv_5,
                   verbose = FALSE,
                   tuneGrid = gbm_grid)
```


```r
plot_grid = expand.grid(
  X1 = seq(min(sim_tst$X1), max(sim_tst$X1), by = 0.01),
  X2 = seq(min(sim_tst$X2), max(sim_tst$X2), by = 0.01)
)

tree_pred = predict(sim_tree_cv, plot_grid)
rf_pred   = predict(sim_rf_oob, plot_grid)
gbm_pred  = predict(sim_gbm_cv, plot_grid)
```


```r
par(mfrow = c(1, 3))
plot(plot_grid$X1, plot_grid$X2, col = tree_pred,
     xlab = "X1", ylab = "X2", pch = 20, main = "Single Tree")
plot(plot_grid$X1, plot_grid$X2, col = rf_pred,
     xlab = "X1", ylab = "X2", pch = 20, main = "Random Forest")
plot(plot_grid$X1, plot_grid$X2, col = gbm_pred,
     xlab = "X1", ylab = "X2", pch = 20, main = "Boosted Trees")
```

![](20-ensemble_files/figure-latex/unnamed-chunk-36-1.pdf)<!-- --> 


## External Links

- [Classification and Regression by `randomForest`](http://www.bios.unc.edu/~dzeng/BIOS740/randomforest.pdf) - Introduction to the `randomForest` package in `R` news.
- [`ranger`: A Fast Implementation of Random Forests](https://github.com/imbs-hl/ranger) - Alternative package for fitting random forests with potentially better speed.
- [On `ranger`'s respect.unordered.factors Argument](http://www.win-vector.com/blog/2016/05/on-ranger-respect-unordered-factors/) - A note on handling of categorical variables with random forests.
- [Extremely Randomized Trees](https://pdfs.semanticscholar.org/336a/165c17c9c56160d332b9f4a2b403fccbdbfb.pdf)
- [`extraTrees` Method for Classificationand Regression](https://cran.r-project.org/web/packages/extraTrees/vignettes/extraTrees.pdf)
- [XGBoost](http://xgboost.readthedocs.io/en/latest/) - Scalable and Flexible Gradient Boosting
- [XGBoost `R` Tutorial](http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html)


## RMarkdown

The RMarkdown file for this chapter can be found [**here**](20-ensemble.Rmd). The file was created using `R` version 3.3.3 and the following packages:

- Base Packages, Attached


```
## [1] "methods"   "parallel"  "splines"   "stats"     "graphics"  "grDevices"
## [7] "utils"     "datasets"  "base"
```

- Additional Packages, Attached


```
##  [1] "rpart.plot"   "rpart"        "mlbench"      "plyr"        
##  [5] "caret"        "ggplot2"      "gbm"          "lattice"     
##  [9] "survival"     "randomForest" "ISLR"         "MASS"        
## [13] "tree"
```

- Additional Packages, Not Attached


```
##  [1] "Rcpp"         "compiler"     "nloptr"       "class"       
##  [5] "iterators"    "tools"        "lme4"         "digest"      
##  [9] "evaluate"     "tibble"       "gtable"       "nlme"        
## [13] "mgcv"         "Matrix"       "foreach"      "yaml"        
## [17] "SparseM"      "e1071"        "stringr"      "knitr"       
## [21] "MatrixModels" "stats4"       "nnet"         "rprojroot"   
## [25] "grid"         "rmarkdown"    "bookdown"     "minqa"       
## [29] "reshape2"     "car"          "magrittr"     "backports"   
## [33] "scales"       "codetools"    "ModelMetrics" "htmltools"   
## [37] "pbkrtest"     "colorspace"   "quantreg"     "stringi"     
## [41] "lazyeval"     "munsell"
```






