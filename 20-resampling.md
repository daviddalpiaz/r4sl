# Resampling



- **NOTE**: This chapter is currently be re-written and will likely change considerably in the near future. It is currently lacking in a number of ways mostly narrative.

In this chapter we introduce **resampling** methods, in particular **cross-validation**. We will highlight the need for cross-validation by comparing it to our previous approach, which was to simply set aside some "test" data that we used for evaluating a model fit using "training" data. We will now refer to these held-out samples as the **validation** data and this approach as the **validation set approach**. Along the way we'll redefine the notion of a "test" dataset.

To illustrate the use of resampling techniques, we'll consider a regression setup with a single predictor $x$, and a regression function $f(x) = x^3$. Adding an additional noise parameter, we define the entire data generating process as

$$
Y \sim N(\mu = x^3, \sigma^2 = 0.25 ^ 2)
$$

We write an `R` function that generates datasets according to this process.


```r
gen_sim_data = function(sample_size) {
  x = runif(n = sample_size, min = -1, max = 1)
  y = rnorm(n = sample_size, mean = x ^ 3, sd = 0.25)
  data.frame(x, y)
}
```

We first simulate a single dataset, which we also split into a *train* and *validation* set. Here, the validation set is 20% of the data.


```r
set.seed(42)
sim_data = gen_sim_data(sample_size = 200)
sim_idx  = sample(1:nrow(sim_data), 160)
sim_trn  = sim_data[sim_idx, ]
sim_val  = sim_data[-sim_idx, ]
```

We plot this training data, as well as the true regression function.


```r
plot(y ~ x, data = sim_trn, col = "dodgerblue", pch = 20)
grid()
curve(x ^ 3, add = TRUE, col = "black", lwd = 2)
```



\begin{center}\includegraphics{20-resampling_files/figure-latex/unnamed-chunk-3-1} \end{center}


```r
calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
```

Recall that we needed this validation set because the training error was far too optimistic for highly flexible models. This would lead us to always use the most flexible model.


```r
fit = lm(y ~ poly(x, 10), data = sim_trn)

calc_rmse(actual = sim_trn$y, predicted = predict(fit, sim_trn))
```

```
## [1] 0.2262618
```

```r
calc_rmse(actual = sim_val$y, predicted = predict(fit, sim_val))
```

```
## [1] 0.2846442
```


## Validation-Set Approach

- TODO: consider fitting polynomial models of degree k = 1:10 to data from this data generating process
- TODO: here, we can consider k, the polynomial degree, as a tuning parameter
- TODO: perform simulation study to evaluate how well validation set approach works


```r
num_sims = 100
num_degrees = 10
val_rmse = matrix(0, ncol = num_degrees, nrow = num_sims)
```

- TODO: each simulation we will...


```r
set.seed(42)
for (i in 1:num_sims) {
  # simulate data
  sim_data = gen_sim_data(sample_size = 200)
  # set aside validation set
  sim_idx = sample(1:nrow(sim_data), 160)
  sim_trn = sim_data[sim_idx, ]
  sim_val = sim_data[-sim_idx, ]
  # fit models and store RMSEs
  for (j in 1:num_degrees) {
    #fit model
    fit = glm(y ~ poly(x, degree = j), data = sim_trn)
    # calculate error
    val_rmse[i, j] = calc_rmse(actual = sim_val$y, predicted = predict(fit, sim_val))
  }
}
```


\begin{center}\includegraphics{20-resampling_files/figure-latex/unnamed-chunk-8-1} \end{center}

- TODO: issues are hard to "see" but have to do with variability


## Cross-Validation

Instead of using a single test-train split, we instead look to use $K$-fold cross-validation.

$$
\text{RMSE-CV}_{K} = \sum_{k = 1}^{K} \frac{n_k}{n} \text{RMSE}_k
$$

$$
\text{RMSE}_k = \sqrt{\frac{1}{n_k} \sum_{i \in C_k} \left( y_i - \hat{f}^{-k}(x_i) \right)^2 }
$$

- $n_k$ is the number of observations in fold $k$ 
- $C_k$ are the observations in fold $k$
- $\hat{f}^{-k}()$ is the trained model using the training data without fold $k$

If $n_k$ is the same in each fold, then

$$
\text{RMSE-CV}_{K} = \frac{1}{K}\sum_{k = 1}^{K} \text{RMSE}_k
$$

- TODO: create and add graphic that shows the splitting process
- TODO: Can be used with any metric, MSE, RMSE, class-err, class-acc

There are many ways to perform cross-validation in `R`, depending on the statistical learning method of interest. Some methods, for example `glm()` through `boot::cv.glm()` and `knn()` through `knn.cv()` have cross-validation capabilities built-in. We'll use `glm()` for illustration. First we need to convince ourselves that `glm()` can be used to perform the same tasks as `lm()`.


```r
glm_fit = glm(y ~ poly(x, 3), data = sim_trn)
coef(glm_fit)
```

```
##  (Intercept)  poly(x, 3)1  poly(x, 3)2  poly(x, 3)3 
## -0.005513063  4.153963639 -0.207436179  2.078844572
```

```r
lm_fit  = lm(y ~ poly(x, 3), data = sim_trn)
coef(lm_fit)
```

```
##  (Intercept)  poly(x, 3)1  poly(x, 3)2  poly(x, 3)3 
## -0.005513063  4.153963639 -0.207436179  2.078844572
```

By default, `cv.glm()` will report leave-one-out cross-validation (LOOCV).


```r
sqrt(boot::cv.glm(sim_trn, glm_fit)$delta)
```

```
## [1] 0.2372763 0.2372582
```

We are actually given two values. The first is exactly the LOOCV-MSE. The second is a minor correction that we will not worry about. We take a square root to obtain LOOCV-RMSE.

In practice, we often prefer 5 or 10-fold cross-validation for a number of reason, but often most importantly, for computational efficiency.


```r
sqrt(boot::cv.glm(sim_trn, glm_fit, K = 5)$delta)
```

```
## [1] 0.2392979 0.2384206
```

We repeat the above simulation study, this time performing 5-fold cross-validation. With a total sample size of $n = 200$ each validation set has 40 observations, as did the single validation set in the previous simulations.


```r
cv_rmse = matrix(0, ncol = num_degrees, nrow = num_sims)
```


```r
set.seed(42)
for (i in 1:num_sims) {
  # simulate data, use all data for training
  sim_trn = gen_sim_data(sample_size = 200)
  # fit models and store RMSE
  for (j in 1:num_degrees) {
    #fit model
    fit = glm(y ~ poly(x, degree = j), data = sim_trn)
    # calculate error
    cv_rmse[i, j] = sqrt(boot::cv.glm(sim_trn, fit, K = 5)$delta[1])
  }
}
```


\begin{center}\includegraphics{20-resampling_files/figure-latex/unnamed-chunk-14-1} \end{center}


\begin{tabular}{r|r|r|r|r}
\hline
Polynomial Degree & Mean, Val & SD, Val & Mean, CV & SD, CV\\
\hline
1 & 0.292 & 0.031 & 0.294 & 0.015\\
\hline
2 & 0.293 & 0.031 & 0.295 & 0.015\\
\hline
3 & 0.252 & 0.028 & 0.255 & 0.012\\
\hline
4 & 0.253 & 0.028 & 0.255 & 0.013\\
\hline
5 & 0.254 & 0.028 & 0.256 & 0.013\\
\hline
6 & 0.254 & 0.028 & 0.257 & 0.013\\
\hline
7 & 0.255 & 0.028 & 0.258 & 0.013\\
\hline
8 & 0.256 & 0.029 & 0.258 & 0.013\\
\hline
9 & 0.257 & 0.029 & 0.261 & 0.013\\
\hline
10 & 0.259 & 0.030 & 0.262 & 0.014\\
\hline
\end{tabular}


\begin{center}\includegraphics{20-resampling_files/figure-latex/unnamed-chunk-16-1} \end{center}

- TODO: differences: less variance, better selections


## Test Data

The following example, inspired by The Elements of Statistical Learning, will illustrate the need for a dedicated test set which is **never** used in model training. We do this, if for no other reason, because it gives us a quick sanity check that we have cross-validated correctly. To be specific we will always test-train split the data, then perform cross-validation **within the training data**. 

Essentially, this example will also show how to **not** cross-validate properly. It will also show can example of cross-validated in a classification setting.


```r
calc_err = function(actual, predicted) {
  mean(actual != predicted)
}
```

Consider a binary response $Y$ with equal probability to take values $0$ and $1$.

$$
Y \sim \text{bern}(p = 0.5)
$$

Also consider $p = 10,000$ independent predictor variables, $X_j$, each with a standard normal distribution.

$$
X_j \sim N(\mu = 0, \sigma^2 = 1)
$$

We simulate $n = 100$ observations from this data generating process. Notice that the way we've defined this process, none of the $X_j$ are related to $Y$.


```r
set.seed(42)
n = 200
p = 10000
x = replicate(p, rnorm(n))
y = c(rbinom(n = n, size = 1, prob = 0.5))
full_data = data.frame(y, x)
```

Before attempting to perform cross-validation, we test-train split the data, using half of the available data for each. (In practice, with this little data, it would be hard to justify a separate test dataset, but here we do so to illustrate another point.)


```r
trn_idx  = sample(1:nrow(full_data), trunc(nrow(full_data) * 0.5))
trn_data = full_data[trn_idx,   ]
tst_data = full_data[-trn_idx, ]
```

Now we would like to train a logistic regression model to predict $Y$ using the available predictor data. However, here we have $p > n$, which prevents us from fitting logistic regression. To overcome this issue, we will first attempt to find a subset of relevant predictors. To do so, we'll simply find the predictors that are most correlated with the response.


```r
# find correlation between y and each predictor variable
correlations = apply(trn_data[, -1], 2, cor, y = trn_data$y)
```


\begin{center}\includegraphics{20-resampling_files/figure-latex/unnamed-chunk-21-1} \end{center}

While many of these correlations are small, many very close to zero, some are as large as 0.40. Since our training data has 50 observations, we'll select the 25 predictors with the largest (absolute) correlations.


```r
selected = order(abs(correlations), decreasing = TRUE)[1:25]
correlations[selected]
```

```
##      X2596      X4214      X9335      X8569      X3299      X2533 
##  0.3543596  0.3523432 -0.3479568 -0.3457459 -0.3454538  0.3432992 
##      X2638      X4737      X2542      X8624      X6201      X4186 
## -0.3393733 -0.3314835  0.3228942 -0.3193488  0.3187754 -0.3181454 
##      X7600      X8557      X3273      X5639      X4482      X7593 
##  0.3175957  0.3159638 -0.3117192  0.3113686  0.3109364  0.3094102 
##      X7374      X7283      X9888       X518      X9970      X7654 
##  0.3090942 -0.3086637  0.3069136 -0.3066874 -0.3061039 -0.3042648 
##      X9329 
## -0.3038140
```

We subset the training and test sets to contain only the response as well as these 25 predictors.


```r
trn_screen = trn_data[c(1, selected)]
tst_screen = tst_data[c(1, selected)]
```

Then we finally fit an additive logistic regression using this subset of predictors. We perform 10-fold cross-validation to obtain an estimate of the classification error.


```r
add_log_mod = glm(y ~ ., data = trn_screen, family = "binomial")
boot::cv.glm(trn_screen, add_log_mod, K = 10)$delta[1]
```

```
## [1] 0.3166792
```

The 10-fold cross-validation is suggesting a classification error estimate of almost 30%.


```r
add_log_pred = (predict(add_log_mod, newdata = tst_screen, type = "response") > 0.5) * 1
calc_err(predicted = add_log_pred, actual = tst_screen$y)
```

```
## [1] 0.5
```

However, if we obtain an estimate of the error using the set, we see an error rate of 50%. No better than guessing! But since $Y$ has no relationship with the predictors, this is actually what we would expect. This incorrect method we'll call screen-then-validate.

Now, we will correctly screen-while-validating. Essentially, instead of simply cross-validating the logistic regression, we also need to cross validate the screening process. That is, we won't simply use the same variables for each fold, we get the "best" predictors for each fold.

For methods that do not have a built-in ability to perform cross-validation, or for methods that have limited cross-validation capability, we will need to write our own code for cross-validation. (Spoiler: This is not completely true, but let's pretend it is, so we can see how to perform cross-validation from scratch.)

This essentially amounts to randomly splitting the data, then looping over the splits. The `createFolds()` function from the `caret()` package will make this much easier.


```r
caret::createFolds(trn_data$y, k = 10)
```

```
## $Fold01
##  [1]  2  6 10 28 66 69 70 89 94 98
## 
## $Fold02
##  [1] 27 30 32 33 34 56 74 80 85 96
## 
## $Fold03
##  [1]  8 23 29 31 39 53 57 60 61 72
## 
## $Fold04
##  [1]  9 15 16 21 41 44 54 63 71 99
## 
## $Fold05
##  [1]  5 12 17 51 62 68 81 82 92 97
## 
## $Fold06
##  [1]  7 13 19 40 43 55 75 77 87 90
## 
## $Fold07
##  [1]  18  42  45  47  48  73  83  88  91 100
## 
## $Fold08
##  [1]  4 11 35 37 46 52 64 76 79 84
## 
## $Fold09
##  [1]  1 14 20 22 26 36 50 59 67 78
## 
## $Fold10
##  [1]  3 24 25 38 49 58 65 86 93 95
```



```r
# use the caret package to obtain 10 "folds"
folds = caret::createFolds(trn_data$y, k = 10)

# for each fold
# - pre-screen variables on the 9 training folds
# - fit model to these variables
# - get error on validation fold
fold_err = rep(0, length(folds))

for (i in seq_along(folds)) {

  # split for fold i  
  trn_fold = trn_data[-folds[[i]], ]
  val_fold = trn_data[folds[[i]], ]

  # screening for fold i  
  correlations = apply(trn_fold[, -1], 2, cor, y = trn_fold[,1])
  selected = order(abs(correlations), decreasing = TRUE)[1:25]
  trn_fold_screen = trn_fold[ , c(1, selected)]
  val_fold_screen = val_fold[ , c(1, selected)]

  # error for fold i  
  add_log_mod = glm(y ~ ., data = trn_fold_screen, family = "binomial")
  add_log_prob = predict(add_log_mod, newdata = val_fold_screen, type = "response")
  add_log_pred = ifelse(add_log_prob > 0.5, yes = 1, no = 0)
  fold_err[i] = mean(add_log_pred != val_fold_screen$y)
  
}

# report all 10 validation fold errors
fold_err
```

```
##  [1] 0.5 0.5 0.6 0.5 0.6 0.5 0.7 0.6 0.2 0.2
```

```r
# properly cross-validated error
# this roughly matches what we expect in the test set
mean(fold_err)
```

```
## [1] 0.49
```

- TODO: note that, even cross-validated correctly, this isn't a brilliant variable selection procedure. (it completely ignores interactions and correlations among the predictors. however, if it works, it works.) next chapters...


## Bootstrap

ISL discusses the bootstrap, which is another resampling method. However, it is less relevant to the statistical learning tasks we will encounter. It could be used to replace cross-validation, but encounters significantly more computation.

It could be more useful if we were to attempt to calculate the bias and variance of a prediction (estimate) without access to the data generating process. Return to the bias-variance tradeoff chapter and think about how the bootstrap could be used to obtain estimates of bias and variance with a single dataset, instead of repeated simulated datasets.



## Which $K$?

- TODO: LOO vs 5 vs 10
- TODO: bias and variance



## Summary

- TODO: using cross validation for: tuning, error estimation


## External Links

- [YouTube: Cross-Validation, Part 1](https://www.youtube.com/watch?v=m5StqDv-YlM) - Video from user "mathematicalmonk" which introduces $K$-fold cross-validation in greater detail.
- [YouTube: Cross-Validation, Part 2](https://www.youtube.com/watch?v=OcJwdF8zBjM) - Continuation which discusses selection and resampling strategies.
- [YouTube: Cross-Validation, Part 3](https://www.youtube.com/watch?v=mvbBycl8BNM) - Continuation which discusses choice of $K$.
- [Blog: Fast Computation of Cross-Validation in Linear Models](http://robjhyndman.com/hyndsight/loocv-linear-models/) - Details for using leverage to speed-up LOOCV for linear models.
- [OTexts: Bootstrap](https://www.otexts.org/1467) - Some brief mathematical details of the bootstrap.


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](20-resampling.Rmd). The file was created using `R` version 3.5.2.
