# Ensemble Methods

**Chapter Status:** Currently chapter is rather lacking in narrative and gives no introduction to the theory of the methods. The `R` code is in a reasonable place, but is generally a little heavy on the output, and could use some better summary of results. Using `Boston` for regression seems OK, but would like a better dataset for classification.



In this chapter, we'll consider ensembles of trees.

## Regression

We first consider the regression case, using the `Boston` data from the `MASS` package. We will use RMSE as our metric, so we write a function which will help us along the way.


```r
calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
```

We also load all of the packages that we will need.


```r
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(caret)
library(MASS)
library(ISLR)
```

We first test-train split the data and fit a single tree using `rpart`.


```r
set.seed(18)
boston_idx = sample(1:nrow(Boston), nrow(Boston) / 2)
boston_trn = Boston[boston_idx,]
boston_tst = Boston[-boston_idx,]
```

### Tree Model


```r
boston_tree = rpart(medv ~ ., data = boston_trn)
```


```r
boston_tree_tst_pred = predict(boston_tree, newdata = boston_tst)
plot(boston_tree_tst_pred, boston_tst$medv, 
     xlab = "Predicted", ylab = "Actual", 
     main = "Predicted vs Actual: Single Tree, Test Data",
     col = "dodgerblue", pch = 20)
grid()
abline(0, 1, col = "darkorange", lwd = 2)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-5-1} \end{center}


```r
(tree_tst_rmse = calc_rmse(boston_tree_tst_pred, boston_tst$medv))
```

```
## [1] 5.458088
```

### Linear Model

Last time, we also fit an additive linear model, which we found to work better than the tree. The test RMSE is lower, and the predicted vs actual plot looks much better.


```r
boston_lm = lm(medv ~ ., data = boston_trn)
```


```r
boston_lm_tst_pred = predict(boston_lm, newdata = boston_tst)
plot(boston_lm_tst_pred, boston_tst$medv,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Linear Model, Test Data",
     col = "dodgerblue", pch = 20)
grid()
abline(0, 1, col = "darkorange", lwd = 2)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-8-1} \end{center}


```r
(lm_tst_rmse = calc_rmse(boston_lm_tst_pred, boston_tst$medv))
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
##           Mean of squared residuals: 13.78573
##                     % Var explained: 81.48
```


```r
boston_bag_tst_pred = predict(boston_bag, newdata = boston_tst)
plot(boston_bag_tst_pred,boston_tst$medv,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Bagged Model, Test Data",
     col = "dodgerblue", pch = 20)
grid()
abline(0, 1, col = "darkorange", lwd = 2)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-11-1} \end{center}


```r
(bag_tst_rmse = calc_rmse(boston_bag_tst_pred, boston_tst$medv))
```

```
## [1] 3.844243
```

Here we see two interesting results. First, the predicted versus actual plot no longer has a small number of predicted values. Second, our test error has dropped dramatically. Also note that the "Mean of squared residuals" which is output by `randomForest` is the **Out of Bag** estimate of the error.


```r
plot(boston_bag, col = "dodgerblue", lwd = 2, main = "Bagged Trees: Error vs Number of Trees")
grid()
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-13-1} \end{center}


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
##           Mean of squared residuals: 12.54771
##                     % Var explained: 83.14
```


```r
importance(boston_forest, type = 1)
```

```
##           %IncMSE
## crim    11.159069
## zn       4.736893
## indus    9.167686
## chas     2.826810
## nox     11.040039
## rm      30.787637
## age      8.448608
## dis      9.733887
## rad      4.725829
## tax      8.622564
## ptratio 11.718528
## black    7.061299
## lstat   23.357019
```

```r
varImpPlot(boston_forest, type = 1)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-15-1} \end{center}


```r
boston_forest_tst_pred = predict(boston_forest, newdata = boston_tst)
plot(boston_forest_tst_pred, boston_tst$medv,
     xlab = "Predicted", ylab = "Actual",
     main = "Predicted vs Actual: Random Forest, Test Data",
     col = "dodgerblue", pch = 20)
grid()
abline(0, 1, col = "darkorange", lwd = 2)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-16-1} \end{center}


```r
(forest_tst_rmse = calc_rmse(boston_forest_tst_pred, boston_tst$medv))
```

```
## [1] 3.710097
```

```r
boston_forest_trn_pred = predict(boston_forest, newdata = boston_trn)
forest_trn_rmse = calc_rmse(boston_forest_trn_pred, boston_trn$medv)
forest_oob_rmse = calc_rmse(boston_forest$predicted, boston_trn$medv)
```

Here we note three RMSEs. The training RMSE (which is optimistic), the OOB RMSE (which is a reasonable estimate of the test error) and the test RMSE. Also note that variables importance was calculated.


```
##       Data    Error
## 1 Training 1.573562
## 2      OOB 3.542275
## 3     Test 3.710097
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
tibble::as_tibble(summary(booston_boost))
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-20-1} \end{center}

```
## # A tibble: 13 x 2
##    var     rel.inf
##  * <fct>     <dbl>
##  1 lstat    34.9  
##  2 rm       30.6  
##  3 dis       9.85 
##  4 crim      6.43 
##  5 black     4.43 
##  6 nox       4.30 
##  7 age       3.34 
##  8 ptratio   2.31 
##  9 tax       1.18 
## 10 rad       0.905
## 11 indus     0.883
## 12 chas      0.679
## 13 zn        0.131
```


```r
par(mfrow = c(1, 3))
plot(booston_boost, i = "rm", col = "dodgerblue", lwd = 2)
grid()
plot(booston_boost, i = "lstat", col = "dodgerblue", lwd = 2)
grid()
plot(booston_boost, i = "dis", col = "dodgerblue", lwd = 2)
grid()
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-21-1} \end{center}


```r
boston_boost_tst_pred = predict(booston_boost, newdata = boston_tst, n.trees = 5000)
(boost_tst_rmse = calc_rmse(boston_boost_tst_pred, boston_tst$medv))
```

```
## [1] 3.381453
```


```r
plot(boston_boost_tst_pred, boston_tst$medv,
     xlab = "Predicted", ylab = "Actual", 
     main = "Predicted vs Actual: Boosted Model, Test Data",
     col = "dodgerblue", pch = 20)
grid()
abline(0, 1, col = "darkorange", lwd = 2)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-23-1} \end{center}

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
## 1   Single Tree  5.458088
## 2  Linear Model  5.125877
## 3       Bagging  3.844243
## 4 Random Forest  3.710097
## 5      Boosting  3.381453
```

While a single tree does not beat linear regression, each of the ensemble methods perform much better!


## Classification

We now return to the `Carseats` dataset and the classification setting. We see that an additive logistic regression performs much better than a single tree, but we expect ensemble methods to bring trees closer to the logistic regression. Can they do better?

We now use prediction accuracy as our metric:


```r
calc_acc = function(actual, predicted) {
  mean(actual == predicted)
}
```


```r
data(Carseats)
Carseats$Sales = as.factor(ifelse(Carseats$Sales <= 8, "Low", "High"))
```


```r
set.seed(2)
seat_idx = sample(1:nrow(Carseats), 200)
seat_trn = Carseats[seat_idx,]
seat_tst = Carseats[-seat_idx,]
```


### Tree Model


```r
seat_tree = rpart(Sales ~ ., data = seat_trn)
```


```r
rpart.plot(seat_tree)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-29-1} \end{center}


```r
seat_tree_tst_pred = predict(seat_tree, seat_tst, type = "class")
table(predicted = seat_tree_tst_pred, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   64  26
##      Low    20  90
```

```r
(tree_tst_acc = calc_acc(predicted = seat_tree_tst_pred, actual = seat_tst$Sales))
```

```
## [1] 0.77
```


### Logistic Regression


```r
seat_glm = glm(Sales ~ ., data = seat_trn, family = "binomial")
```


```r
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
(glm_tst_acc = calc_acc(predicted = seat_glm_tst_pred, actual = seat_tst$Sales))
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
##         OOB estimate of  error rate: 20%
## Confusion matrix:
##      High Low class.error
## High   53  27   0.3375000
## Low    13 107   0.1083333
```


```r
seat_bag_tst_pred = predict(seat_bag, newdata = seat_tst)
table(predicted = seat_bag_tst_pred, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   66  21
##      Low    18  95
```

```r
(bag_tst_acc = calc_acc(predicted = seat_bag_tst_pred, actual = seat_tst$Sales))
```

```
## [1] 0.805
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
##         OOB estimate of  error rate: 22.5%
## Confusion matrix:
##      High Low class.error
## High   49  31   0.3875000
## Low    14 106   0.1166667
```


```r
seat_forest_tst_perd = predict(seat_forest, newdata = seat_tst)
table(predicted = seat_forest_tst_perd, actual = seat_tst$Sales)
```

```
##          actual
## predicted High Low
##      High   62  19
##      Low    22  97
```

```r
(forest_tst_acc = calc_acc(predicted = seat_forest_tst_perd, actual = seat_tst$Sales))
```

```
## [1] 0.795
```


### Boosting

To perform boosting, we modify the response to be `0` and `1` to work with `gbm`. Later we will use `caret` to fit `gbm` models, which will avoid this annoyance.


```r
seat_trn_mod = seat_trn
seat_trn_mod$Sales = as.numeric(ifelse(seat_trn_mod$Sales == "Low", "0", "1"))
```


```r
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
##      High   71  18
##      Low    13  98
```

```r
(boost_tst_acc = calc_acc(predicted = seat_boost_tst_pred, actual = seat_tst$Sales))
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
## 3             Bagging        0.805
## 4       Random Forest        0.795
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

To tune a Random Forest in `caret` we will use `method = "rf"` which uses the `randomForest` function in the background. Here we elect to use the OOB training control that we created. We could also use cross-validation, however it will likely select a similar model, but require much more time.

We setup a grid of `mtry` values which include all possible values since there are $10$ predictors in the dataset. An `mtry` of $10$ is actually bagging.


```r
dim(seat_trn)
```

```
## [1] 200  11
```

```r
rf_grid =  expand.grid(mtry = 1:10)
```


```r
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
##    1    0.730     0.3807339
##    2    0.775     0.5033113
##    3    0.795     0.5591398
##    4    0.790     0.5474138
##    5    0.795     0.5610278
##    6    0.790     0.5512821
##    7    0.805     0.5806452
##    8    0.795     0.5628998
##    9    0.785     0.5376344
##   10    0.775     0.5202559
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 7.
```


```r
calc_acc(predict(seat_rf_tune, seat_tst), seat_tst$Sales)
```

```
## [1] 0.795
```

The results returned are based on the OOB samples. (Coincidentally, the test accuracy is the same as the best accuracy found using OOB samples.) Note that when using OOB, for some reason the default plot is not what you would expect and is not at all useful. (Which is why it is omitted here.)


```r
seat_rf_tune$bestTune
```

```
##   mtry
## 7    7
```

Based on these results, we would select the random forest model with an `mtry` of 7. Note that based on the OOB estimates, the bagging model is expected to perform worse than this selected model, however, based on our results above, that is not what we find to be true in our test set.

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

By default, calling `plot` here will produce a nice graphic summarizing the results. 


```r
plot(seat_gbm_tune)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-48-1} \end{center}


```r
calc_acc(predict(seat_gbm_tune, seat_tst), seat_tst$Sales)
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
sim_trn_col = ifelse(sim_trn$class == 1, "darkorange", "dodgerblue")
plot(sim_trn$X1, sim_trn$X2, col = sim_trn_col,
     xlab = "X1", ylab = "X2", main = "Simulated Training Data", pch = 20)
grid()
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-52-1} \end{center}


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


```r
library(rpart.plot)
rpart.plot(sim_tree_cv$finalModel)
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-55-1} \end{center}


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
  X1 = seq(min(sim_tst$X1) - 1, max(sim_tst$X1) + 1, by = 0.01),
  X2 = seq(min(sim_tst$X2) - 1, max(sim_tst$X2) + 1, by = 0.01)
)

tree_pred = predict(sim_tree_cv, plot_grid)
rf_pred   = predict(sim_rf_oob, plot_grid)
gbm_pred  = predict(sim_gbm_cv, plot_grid)

tree_col = ifelse(tree_pred == 1, "darkorange", "dodgerblue")
rf_col   = ifelse(rf_pred == 1, "darkorange", "dodgerblue")
gbm_col  = ifelse(gbm_pred == 1, "darkorange", "dodgerblue")
```


```r
par(mfrow = c(1, 3))
plot(plot_grid$X1, plot_grid$X2, col = tree_col,
     xlab = "X1", ylab = "X2", pch = 20, main = "Single Tree",
     xlim = c(-1, 1), ylim = c(-1, 1))
plot(plot_grid$X1, plot_grid$X2, col = rf_col,
     xlab = "X1", ylab = "X2", pch = 20, main = "Random Forest",
     xlim = c(-1, 1), ylim = c(-1, 1))
plot(plot_grid$X1, plot_grid$X2, col = gbm_col,
     xlab = "X1", ylab = "X2", pch = 20, main = "Boosted Trees",
     xlim = c(-1, 1), ylim = c(-1, 1))
```



\begin{center}\includegraphics{27-ensemble_files/figure-latex/unnamed-chunk-59-1} \end{center}


## External Links

- [Classification and Regression by `randomForest`](http://www.bios.unc.edu/~dzeng/BIOS740/randomforest.pdf) - Introduction to the `randomForest` package in `R` news.
- [`ranger`: A Fast Implementation of Random Forests](https://github.com/imbs-hl/ranger) - Alternative package for fitting random forests with potentially better speed.
- [On `ranger`'s respect.unordered.factors Argument](http://www.win-vector.com/blog/2016/05/on-ranger-respect-unordered-factors/) - A note on handling of categorical variables with random forests.
- [Extremely Randomized Trees](https://pdfs.semanticscholar.org/336a/165c17c9c56160d332b9f4a2b403fccbdbfb.pdf)
- [`extraTrees` Method for Classificationand Regression](https://cran.r-project.org/web/packages/extraTrees/vignettes/extraTrees.pdf)
- [XGBoost](http://xgboost.readthedocs.io/en/latest/) - Scalable and Flexible Gradient Boosting
- [XGBoost `R` Tutorial](http://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html)


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](27-ensemble.Rmd). The file was created using `R` version 3.4.2. The following packages (and their dependencies) were loaded when knitting this file:


```
##  [1] "mlbench"      "ISLR"         "MASS"         "caret"       
##  [5] "ggplot2"      "gbm"          "lattice"      "survival"    
##  [9] "randomForest" "rpart.plot"   "rpart"
```
