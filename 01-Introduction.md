# Introduction

This is the first real chapter.

This is a travis test.

asdf

\BeginKnitrBlock{theorem}\iffalse{-91-80-121-116-104-97-103-111-114-101-97-110-32-84-104-101-111-114-101-109-93-}\fi<div class="theorem"><span class="theorem" id="thm:unnamed-chunk-1"><strong>(\#thm:unnamed-chunk-1) \iffalse (Pythagorean Theorem) \fi </strong></span>For a right triangle, if $c$ denotes the length of the hypotenuse
and $a$ and $b$ denote the lengths of the other two sides, we have
$$a^2 + b^2 = c^2$$</div>\EndKnitrBlock{theorem}

test gitpages build status




```r
install.packages(c("rmarkdown", "tidyverse", "knitr", "ISLR", "caret", 
                   "AppliedPredictiveModeling", "ellipse", "nnet", "pROC", 
                   "knitr", "randomForest", "leaps", "glmnet", "mxnet", "gam", 
                   "tree", "rpart", "gbm", "extraTrees", "kernlab", "e1071",
                   "extraTrees", "sparcl", "formatR"))
```



```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(71)
iris.rf <- randomForest(Species ~ .,
                        data = iris,
                        importance = TRUE,
                        proximity = TRUE)
iris.rf
```

```
## 
## Call:
##  randomForest(formula = Species ~ ., data = iris, importance = TRUE,      proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 5.33%
## Confusion matrix:
##            setosa versicolor virginica class.error
## setosa         50          0         0        0.00
## versicolor      0         46         4        0.08
## virginica       0          4        46        0.08
```

