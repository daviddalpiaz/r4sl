# Generative Models

In this chapter, we continue our discussion of classification methods. We introduce three new methods, each a **generative** method. This in comparison to logistic regression, which is a **discriminative** method.

Generative methods model the joint probability, $p(x, y)$, often by assuming some distribution for the conditional distribution of $X$ given $Y$, $f(x \mid y)$. Bayes theorem is then applied to classify according to $p(y \mid x)$. Discriminative methods directly model this conditional, $p(y \mid x)$. A detailed discussion and analysis can be found in [Ng and Jordan, 2002](https://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes.pdf).

Each of the methods in this chapter will use Bayes theorem to build a classifier.

$$
p_k(x) = P(Y = k \mid X = x) = \frac{\pi_k \cdot f_k(x)}{\sum_{g = 1}^{G} \pi_g \cdot f_g(x)}
$$

We call $p_k(x)$ the **posterior** probability, which we will estimate then use to create classifications. The $\pi_g$ are called the **prior** probabilities for each possible class $g$. That is, $\pi_g = P(Y = g)$, unconditioned on $X$. The $f_g(x)$ are called the **likelihoods**, which are indexed by $g$ to denote that they are conditional on the classes. The denominator is often referred to as a **normalizing constant**.

The methods will differ by placing different modeling assumptions on the likelihoods, $f_g(x)$. For each method, the priors could be learned from data or pre-specified.

For each method, classifications are made to the class with the highest estimated posterior probability, which is equivalent to the class with the largest

$$
\log(\hat{\pi}_k \cdot \hat{f}_k({\mathbf x})).
$$

By substituting the corresponding likelihoods, simplifying, and eliminating unnecessary terms, we could derive the discriminant function for each.

To illustrate these new methods, we return to the iris data, which you may remember has three classes. After a test-train split, we create a number of plots to refresh our memory.


```r
set.seed(430)
iris_obs = nrow(iris)
iris_idx = sample(iris_obs, size = trunc(0.50 * iris_obs))
# iris_index = sample(iris_obs, size = trunc(0.10 * iris_obs))
iris_trn = iris[iris_idx, ]
iris_tst = iris[-iris_idx, ]
```



```r
caret::featurePlot(x = iris_trn[, c("Sepal.Length", "Sepal.Width", 
                                    "Petal.Length", "Petal.Width")], 
                   y = iris_trn$Species,
                   plot = "density", 
                   scales = list(x = list(relation = "free"), 
                                 y = list(relation = "free")), 
                   adjust = 1.5, 
                   pch = "|", 
                   layout = c(2, 2), 
                   auto.key = list(columns = 3))
```

![](11-generative_files/figure-latex/unnamed-chunk-2-1.pdf)<!-- --> 


```r
caret::featurePlot(x = iris_trn[, c("Sepal.Length", "Sepal.Width", 
                                    "Petal.Length", "Petal.Width")], 
                   y = iris_trn$Species,
                   plot = "ellipse",
                   auto.key = list(columns = 3))
```

![](11-generative_files/figure-latex/unnamed-chunk-3-1.pdf)<!-- --> 



```r
caret::featurePlot(x = iris_trn[, c("Sepal.Length", "Sepal.Width", 
                                    "Petal.Length", "Petal.Width")], 
                   y = iris_trn$Species,
                   plot = "box",
                   scales = list(y = list(relation = "free"),
                                 x = list(rot = 90)),
                   layout = c(4, 1))
```

![](11-generative_files/figure-latex/unnamed-chunk-4-1.pdf)<!-- --> 

Especially based on the pairs plot, we see that it should not be too difficult to find a good classifier.

Notice that we use `caret::featurePlot` to access the `featurePlot()` function without loading the entire `caret` package.


## Linear Discriminant Analysis

LDA assumes that the predictors are multivariate normal conditioned on the classes.

$$
X \mid Y = k \sim N(\mu_k, \Sigma)
$$

$$
f_k({\mathbf x}) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}\exp\left[-\frac{1}{2}(\mathbf x - \mu_k)^{\prime}\Sigma^{-1}(\mathbf x - \mu_k)\right]
$$

Notice that $\Sigma$ does **not** depend on $k$, that is, we are assuming the same $\Sigma$ for each class. We then use information from all the classes to estimate $\Sigma$.

To fit an LDA model, we use the `lda()` function from the `MASS` package.


```r
library(MASS)
iris_lda = lda(Species ~ ., data = iris_trn)
iris_lda
```

```
## Call:
## lda(Species ~ ., data = iris_trn)
## 
## Prior probabilities of groups:
##     setosa versicolor  virginica 
##  0.3866667  0.2933333  0.3200000 
## 
## Group means:
##            Sepal.Length Sepal.Width Petal.Length Petal.Width
## setosa         4.958621    3.420690     1.458621    0.237931
## versicolor     6.063636    2.845455     4.318182    1.354545
## virginica      6.479167    2.937500     5.479167    2.045833
## 
## Coefficients of linear discriminants:
##                     LD1        LD2
## Sepal.Length  0.7394386 -0.6107043
## Sepal.Width   1.8232273  1.6011748
## Petal.Length -2.1304796 -1.2400672
## Petal.Width  -2.8835695  3.8585321
## 
## Proportion of trace:
##   LD1   LD2 
## 0.994 0.006
```

Here we see the estimated $\hat{\pi}_k$ and $\hat{\mu}_k$ for each class.


```r
is.list(predict(iris_lda, iris_trn))
```

```
## [1] TRUE
```

```r
names(predict(iris_lda, iris_trn))
```

```
## [1] "class"     "posterior" "x"
```

```r
head(predict(iris_lda, iris_trn)$class, n = 10)
```

```
##  [1] setosa     versicolor versicolor setosa     virginica  versicolor
##  [7] versicolor virginica  setosa     versicolor
## Levels: setosa versicolor virginica
```

```r
head(predict(iris_lda, iris_trn)$posterior, n = 10)
```

```
##           setosa   versicolor    virginica
## 1   1.000000e+00 9.098758e-23 1.863931e-43
## 92  2.858466e-22 9.988462e-01 1.153843e-03
## 77  9.196834e-24 9.988760e-01 1.124025e-03
## 38  1.000000e+00 4.555854e-24 2.290761e-45
## 108 4.462001e-43 3.898154e-04 9.996102e-01
## 83  5.876068e-17 9.999948e-01 5.158096e-06
## 64  5.167661e-24 9.959455e-01 4.054516e-03
## 110 6.922158e-46 8.001878e-07 9.999992e-01
## 30  1.000000e+00 2.473394e-17 4.662999e-36
## 94  8.280923e-15 9.999995e-01 5.000679e-07
```

As we should come to expect, the `predict()` function operates in a new way when called on an `lda` object. By default, it returns an entire list. Within that list `class` stores the classifications and `posterior` contains the estimated probability for each class.


```r
iris_lda_trn_pred = predict(iris_lda, iris_trn)$class
iris_lda_tst_pred = predict(iris_lda, iris_tst)$class
```

We store the predictions made on the train and test sets.


```r
calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}
```


```r
calc_class_err(predicted = iris_lda_trn_pred, actual = iris_trn$Species)
```

```
## [1] 0.02666667
```

```r
calc_class_err(predicted = iris_lda_tst_pred, actual = iris_tst$Species)
```

```
## [1] 0.01333333
```

As expected, LDA performs well on both the train and test data.


```r
table(predicted = iris_lda_tst_pred, actual = iris_tst$Species)
```

```
##             actual
## predicted    setosa versicolor virginica
##   setosa         21          0         0
##   versicolor      0         27         0
##   virginica       0          1        26
```

Looking at the test set, we see that we are perfectly predicting both setosa and versicolor. The only error is labeling a virginica as a versicolor.


```r
iris_lda_flat = lda(Species ~ ., data = iris_trn, prior = c(1, 1, 1) / 3)
iris_lda_flat
```

```
## Call:
## lda(Species ~ ., data = iris_trn, prior = c(1, 1, 1)/3)
## 
## Prior probabilities of groups:
##     setosa versicolor  virginica 
##  0.3333333  0.3333333  0.3333333 
## 
## Group means:
##            Sepal.Length Sepal.Width Petal.Length Petal.Width
## setosa         4.958621    3.420690     1.458621    0.237931
## versicolor     6.063636    2.845455     4.318182    1.354545
## virginica      6.479167    2.937500     5.479167    2.045833
## 
## Coefficients of linear discriminants:
##                     LD1        LD2
## Sepal.Length  0.7403226 -0.6096323
## Sepal.Width   1.8209056  1.6038146
## Petal.Length -2.1286808 -1.2431524
## Petal.Width  -2.8891565  3.8543505
## 
## Proportion of trace:
##   LD1   LD2 
## 0.993 0.007
```

Instead of learning (estimating) the proportion of the three species from the data, we could instead specify them ourselves. Here we choose a uniform distributions over the possible species. We would call this a "flat" prior.


```r
iris_lda_flat_trn_pred = predict(iris_lda_flat, iris_trn)$class
iris_lda_flat_tst_pred = predict(iris_lda_flat, iris_tst)$class
```


```r
calc_class_err(predicted = iris_lda_flat_trn_pred, actual = iris_trn$Species)
```

```
## [1] 0.02666667
```

```r
calc_class_err(predicted = iris_lda_flat_tst_pred, actual = iris_tst$Species)
```

```
## [1] 0.01333333
```

This actually gives a better test accuracy!


## Quadratic Discriminant Analysis

QDA also assumes that the predictors are multivariate normal conditioned on the classes.

$$
X \mid Y = k \sim N(\mu_k, \Sigma_k)
$$

$$
f_k({\mathbf x}) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}}\exp\left[-\frac{1}{2}(\mathbf x - \mu_k)^{\prime}\Sigma_{k}^{-1}(\mathbf x - \mu_k)\right]
$$

Notice that now $\Sigma_k$ **does** depend on $k$, that is, we are allowing a different $\Sigma_k$ for each class. We only use information from class $k$ to estimate $\Sigma_k$. 


```r
iris_qda = qda(Species ~ ., data = iris_trn)
iris_qda
```

```
## Call:
## qda(Species ~ ., data = iris_trn)
## 
## Prior probabilities of groups:
##     setosa versicolor  virginica 
##  0.3866667  0.2933333  0.3200000 
## 
## Group means:
##            Sepal.Length Sepal.Width Petal.Length Petal.Width
## setosa         4.958621    3.420690     1.458621    0.237931
## versicolor     6.063636    2.845455     4.318182    1.354545
## virginica      6.479167    2.937500     5.479167    2.045833
```

Here the output is similar to LDA, again giving the estimated $\hat{\pi}_k$ and $\hat{\mu}_k$ for each class. Like `lda()`, the `qda()` function is found in the `MASS` package.

Consider trying to fit QDA again, but this time with a smaller training set. (Use the commented line above to obtain a smaller test set.) This will cause an error because there are not enough observations within each class to estimate the large number of parameters in the $\Sigma_k$ matrices. This is less of a problem with LDA, since all observations, no matter the class, are being use to estimate the shared $\Sigma$ matrix.


```r
iris_qda_trn_pred = predict(iris_qda, iris_trn)$class
iris_qda_tst_pred = predict(iris_qda, iris_tst)$class
```

The `predict()` function operates the same as the `predict()` function for LDA.


```r
calc_class_err(predicted = iris_qda_trn_pred, actual = iris_trn$Species)
```

```
## [1] 0.01333333
```

```r
calc_class_err(predicted = iris_qda_tst_pred, actual = iris_tst$Species)
```

```
## [1] 0.05333333
```


```r
table(predicted = iris_qda_tst_pred, actual = iris_tst$Species)
```

```
##             actual
## predicted    setosa versicolor virginica
##   setosa         21          0         0
##   versicolor      0         25         1
##   virginica       0          3        25
```

Here we find that QDA is not performing as well as LDA. It is misclassifying versicolors. Since QDA is a more complex model than LDA (many more parameters) we would say that QDA is overfitting here.

Also note that, QDA creates quadratic decision boundaries, while LDA creates linear decision boundaries. We could also add quadratic terms to LDA to allow it to create quadratic decision boundaries.


## Naive Bayes

Naive Bayes comes in many forms. With only numeric predictors, it often assumes a multivariate normal conditioned on the classes, but a very specific multivariate normal.

$$
{\mathbf X} \mid Y = k \sim N(\mu_k, \Sigma_k)
$$

Naive Bayes assumes that the predictors $X_1, X_2, \ldots, X_p$ are independent. This is the "naive" part of naive Bayes. The Bayes part is nothing new. Since $X_1, X_2, \ldots, X_p$ are assumed independent, each $\Sigma_k$ is diagonal, that is, we assume no correlation between predictors. Independence implies zero correlation.

This will allow us to write the (joint) likelihood as a product of univariate distributions. In this case, the product of univariate normal distributions instead of a (joint) multivariate distribution.

$$
f_k(x) = \prod_{j = 1}^{j = p} f_{kj}(x_j)
$$

Here, $f_{kj}(x_j)$ is the density for the $j$-th predictor conditioned on the $k$-th class. Notice that there is a $\sigma_{kj}$ for each predictor for each class.

$$
f_{kj}(x_j) = \frac{1}{\sigma_{kj}\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{x_j - \mu_{kj}}{\sigma_{kj}}\right)^2\right]
$$

When $p = 1$, this version of naive Bayes is equivalent to QDA.


```r
library(e1071)
iris_nb = naiveBayes(Species ~ ., data = iris_trn)
iris_nb
```

```
## 
## Naive Bayes Classifier for Discrete Predictors
## 
## Call:
## naiveBayes.default(x = X, y = Y, laplace = laplace)
## 
## A-priori probabilities:
## Y
##     setosa versicolor  virginica 
##  0.3866667  0.2933333  0.3200000 
## 
## Conditional probabilities:
##             Sepal.Length
## Y                [,1]      [,2]
##   setosa     4.958621 0.3212890
##   versicolor 6.063636 0.5636154
##   virginica  6.479167 0.5484993
## 
##             Sepal.Width
## Y                [,1]      [,2]
##   setosa     3.420690 0.4012296
##   versicolor 2.845455 0.3262007
##   virginica  2.937500 0.3267927
## 
##             Petal.Length
## Y                [,1]      [,2]
##   setosa     1.458621 0.1880677
##   versicolor 4.318182 0.5543219
##   virginica  5.479167 0.4995469
## 
##             Petal.Width
## Y                [,1]       [,2]
##   setosa     0.237931 0.09788402
##   versicolor 1.354545 0.21979920
##   virginica  2.045833 0.29039578
```

Many packages implement naive Bayes. Here we choose to use `naiveBayes()` from the package `e1071`. (The name of this package has an interesting history. Based on the name you wouldn't know it, but the package contains many functions related to machine learning.)

The `Conditional probabilities:` portion of the output gives the mean and standard deviation of the normal distribution for each predictor in each class. Notice how these mean estimates match those for LDA and QDA above.

Note that `naiveBayes()` will work without a factor response, but functions much better with one. (Especially when making predictions.) If you are using a `0` and `1` response, you might consider coercing to a factor first.


```r
head(predict(iris_nb, iris_trn))
```

```
## [1] setosa     versicolor versicolor setosa     virginica  versicolor
## Levels: setosa versicolor virginica
```

```r
head(predict(iris_nb, iris_trn, type = "class"))
```

```
## [1] setosa     versicolor versicolor setosa     virginica  versicolor
## Levels: setosa versicolor virginica
```

```r
head(predict(iris_nb, iris_trn, type = "raw"))
```

```
##             setosa   versicolor    virginica
## [1,]  1.000000e+00 3.096444e-15 5.172277e-27
## [2,]  1.079241e-93 9.833098e-01 1.669021e-02
## [3,] 6.378471e-106 9.210439e-01 7.895614e-02
## [4,]  1.000000e+00 1.691578e-16 2.882941e-28
## [5,] 1.791407e-209 3.462703e-04 9.996537e-01
## [6,]  4.538228e-59 9.999316e-01 6.835677e-05
```

Oh look, `predict()` has another new mode of operation. If only there were a way to unify the `predict()` function across all of these methods...


```r
iris_nb_trn_pred = predict(iris_nb, iris_trn)
iris_nb_tst_pred = predict(iris_nb, iris_tst)
```


```r
calc_class_err(predicted = iris_nb_trn_pred, actual = iris_trn$Species)
```

```
## [1] 0.05333333
```

```r
calc_class_err(predicted = iris_nb_tst_pred, actual = iris_tst$Species)
```

```
## [1] 0.02666667
```


```r
table(predicted = iris_nb_tst_pred, actual = iris_tst$Species)
```

```
##             actual
## predicted    setosa versicolor virginica
##   setosa         21          0         0
##   versicolor      0         28         2
##   virginica       0          0        24
```

Like LDA, naive Bayes is having trouble with virginica. 




\begin{tabular}{l|r|r}
\hline
Method & Train Error & Test Error\\
\hline
LDA & 0.0266667 & 0.0133333\\
\hline
LDA, Flat Prior & 0.0266667 & 0.0133333\\
\hline
QDA & 0.0133333 & 0.0533333\\
\hline
Naive Bayes & 0.0533333 & 0.0266667\\
\hline
\end{tabular}

Summarizing the results, we see that Naive Bayes is the worst of LDA, QDA, and NB for this data. So why should we care about naive Bayes?

The strength of naive Bayes comes from its ability to handle a large number of predictors, $p$, even with a limited sample size $n$. Even with the naive independence assumption, naive Bayes works rather well in practice. Also because of this assumption, we can often train naive Bayes where LDA and QDA may be impossible to train because of the large number of parameters relative to the number of observations.

Here naive Bayes doesn't get a chance to show its strength since LDA and QDA already perform well, and the number of predictors is low. The choice between LDA and QDA is mostly down to a consideration about the amount of complexity needed.


## Discrete Inputs

So far, we have assumed that all predictors are numeric. What happens with categorical predictors?


```r
iris_trn_mod = iris_trn

iris_trn_mod$Sepal.Width = ifelse(iris_trn$Sepal.Width > 3, 
                                  ifelse(iris_trn$Sepal.Width > 4, 
                                         "Large", "Medium"),
                                  "Small")

unique(iris_trn_mod$Sepal.Width)
```

```
## [1] "Medium" "Small"  "Large"
```

Here we make a new dataset where `Sepal.Width` is categorical, with levels `Small`, `Medium`, and `Large`. We then try to train classifiers using only the sepal variables.


```r
naiveBayes(Species ~ Sepal.Length + Sepal.Width, data = iris_trn_mod)
```

```
## 
## Naive Bayes Classifier for Discrete Predictors
## 
## Call:
## naiveBayes.default(x = X, y = Y, laplace = laplace)
## 
## A-priori probabilities:
## Y
##     setosa versicolor  virginica 
##  0.3866667  0.2933333  0.3200000 
## 
## Conditional probabilities:
##             Sepal.Length
## Y                [,1]      [,2]
##   setosa     4.958621 0.3212890
##   versicolor 6.063636 0.5636154
##   virginica  6.479167 0.5484993
## 
##             Sepal.Width
## Y                 Large     Medium      Small
##   setosa     0.06896552 0.75862069 0.17241379
##   versicolor 0.00000000 0.27272727 0.72727273
##   virginica  0.00000000 0.33333333 0.66666667
```

Naive Bayes makes a somewhat obvious and intelligent choice to model the categorical variable as a multinomial. It then estimates the probability parameters of a multinomial distribution.


```r
lda(Species ~ Sepal.Length + Sepal.Width, data = iris_trn_mod)
```

```
## Call:
## lda(Species ~ Sepal.Length + Sepal.Width, data = iris_trn_mod)
## 
## Prior probabilities of groups:
##     setosa versicolor  virginica 
##  0.3866667  0.2933333  0.3200000 
## 
## Group means:
##            Sepal.Length Sepal.WidthMedium Sepal.WidthSmall
## setosa         4.958621         0.7586207        0.1724138
## versicolor     6.063636         0.2727273        0.7272727
## virginica      6.479167         0.3333333        0.6666667
## 
## Coefficients of linear discriminants:
##                        LD1        LD2
## Sepal.Length      2.194825  0.7108153
## Sepal.WidthMedium 1.296250 -0.7224618
## Sepal.WidthSmall  2.922089 -2.5286497
## 
## Proportion of trace:
##    LD1    LD2 
## 0.9929 0.0071
```

LDA however creates dummy variables, here with `Large` is the reference level, then continues to model them as normally distributed. Not great, but better then not using a categorical variable.


## `rmarkdown`

The `rmarkdown` file for this chapter can be found [**here**](11-generative.Rmd). The file was created using `R` version 4.0.2. The following packages (and their dependencies) were loaded when knitting this file:


```
## [1] "e1071" "MASS"
```
