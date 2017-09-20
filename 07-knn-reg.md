# $k$-Nearest Neighbors {#knn-reg}

## $k$-Nearest Neighbors

$$
\hat{f}(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x, \mathcal{D})} y_i
$$

## KNN in `R`


```r
library(FNN)
library(MASS)
data(Boston)
```


```r
set.seed(42)
boston_idx = sample(1:nrow(Boston), size = 250)
trn_boston = Boston[boston_idx, ]
tst_boston  = Boston[-boston_idx, ]
```


```r
X_trn_boston = trn_boston["lstat"]
X_tst_boston = tst_boston["lstat"]
y_trn_boston = trn_boston["medv"]
y_tst_boston = tst_boston["medv"]
```

We create an additional "test" set `lstat_grid`, that is a grid of `lstat` values at which we will predict `medv` in order to create graphics.


```r
X_trn_boston_min = min(X_trn_boston)
X_trn_boston_max = max(X_trn_boston)
lstat_grid = data.frame(lstat = seq(X_trn_boston_min, X_trn_boston_max, 
                                    by = 0.01))
```

To perform KNN for regression, we will need `knn.reg()` from the `FNN` package. Notice that, we do **not** load this package, but instead use `FNN::knn.reg` to access the function. Note that, in the future, we'll need to be careful about loading the `FNN` package as it also contains a function called `knn`. This function also appears in the `class` package which we will likely use later.


```r
pred_001 = knn.reg(train = X_trn_boston, test = lstat_grid, y = y_trn_boston, k = 1)
pred_005 = knn.reg(train = X_trn_boston, test = lstat_grid, y = y_trn_boston, k = 5)
pred_010 = knn.reg(train = X_trn_boston, test = lstat_grid, y = y_trn_boston, k = 10)
pred_050 = knn.reg(train = X_trn_boston, test = lstat_grid, y = y_trn_boston, k = 50)
pred_100 = knn.reg(train = X_trn_boston, test = lstat_grid, y = y_trn_boston, k = 100)
pred_250 = knn.reg(train = X_trn_boston, test = lstat_grid, y = y_trn_boston, k = 250)
```

We make predictions for various values of `k`. Note that `250` is the total number of observations in this training dataset.

![](07-knn-reg_files/figure-latex/unnamed-chunk-6-1.pdf)<!-- --> 

We see that `k = 1` is clearly overfitting, as `k = 1` is a very complex, highly variable model. Conversely, `k = 250` is clearly underfitting the data, as `k = 250` is a very simple, low variance model. In fact, here it is predicting a simple average of all the data at each point.


## Choosing $k$


```r
# calculate train RMSE
# calculate test RMSE
```


## Scaling Data


```r
sim_knn_data = function(n_obs = 50) {
  x1 = seq(0, 10, length.out = n_obs)
  x2 = runif(n = n_obs, min = 0, max = 2)
  x3 = runif(n = n_obs, min = 0, max = 1)
  x4 = runif(n = n_obs, min = 0, max = 5)
  x5 = runif(n = n_obs, min = 0, max = 5)
  y = x1 ^ 2 + rnorm(n = n_obs)
  data.frame(y, x1, x2, x3,x4, x5)
}
```


```r
set.seed(42)
knn_data = sim_knn_data()
```

![](07-knn-reg_files/figure-latex/unnamed-chunk-10-1.pdf)<!-- --> 


## Curse of Dimensionality


```r
set.seed(42)
knn_data_trn = sim_knn_data()
knn_data_tst = sim_knn_data()
```


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](07-knn-reg.Rmd). The file was created using `R` version 3.4.1. The following packages (and their dependencies) were loaded when knitting this file:


```
## [1] "MASS" "FNN"
```
