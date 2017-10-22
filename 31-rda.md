# Regularized Discriminant Analysis

**TOOD:** This chpater is currently empty to reduce build time.

<!-- We now use the  `Sonar` dataset from the `mlbench` package to explore a new regularization method, **regularized discriminant analysis** (RDA), which combines the LDA and QDA. This is similar to how elastic net combines the ridge and lasso. -->

<!-- ## Sonar Data -->

<!-- ```{r} -->
<!-- # this is a temporary workaround for an issue with glmnet, Matrix, and R version 3.3.3 -->
<!-- # see here: http://stackoverflow.com/questions/43282720/r-error-in-validobject-object-when-running-as-script-but-not-in-console -->
<!-- library(methods) -->
<!-- ``` -->

<!-- ```{r, message = FALSE, warning = FALSE} -->
<!-- library(mlbench) -->
<!-- library(caret) -->
<!-- library(glmnet) -->
<!-- library(klaR) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- data(Sonar) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- #View(Sonar) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- table(Sonar$Class) / nrow(Sonar) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- ncol(Sonar) - 1 -->
<!-- ``` -->

<!-- ## RDA -->

<!-- Regularized discriminant analysis uses the same general setup as LDA and QDA but estimates the covariance in a new way, which combines the covariance of QDA $(\hat{\Sigma}_k)$ with the covariance of LDA $(\hat{\Sigma})$ using a tuning parameter $\lambda$. -->

<!-- $$ -->
<!-- \hat{\Sigma}_k(\lambda) = (1-\lambda)\hat{\Sigma}_k + \lambda \hat{\Sigma} -->
<!-- $$ -->

<!-- Using the `rda()` function from the `klaR` package, which `caret` utilizes, makes an additional modification to the covariance matrix, which also has a tuning parameter $\gamma$. -->

<!-- $$ -->
<!-- \hat{\Sigma}_k(\lambda,\gamma) = (1 -\gamma) \hat{\Sigma}_k(\lambda) + \gamma \frac{1}{p} \text{tr}(\hat{\Sigma}_k(\lambda)) I -->
<!-- $$ -->

<!-- Both $\gamma$ and $\lambda$ can be thought of as mixing parameters, as they both take values between 0 and 1. For the four extremes of $\gamma$ and $\lambda$, the covariance structure reduces to special cases: -->

<!-- - $(\gamma=0, \lambda=0)$: QDA - individual covariance for each group. -->
<!-- - $(\gamma=0, \lambda=1)$: LDA - a common covariance matrix. -->
<!-- - $(\gamma=1, \lambda=0)$: Conditional independent variables - similar to Naive Bayes, but variable variances within group (main diagonal elements) are all equal. -->
<!-- - $(\gamma=1, \lambda=1)$: Classification using euclidean distance - as in previous case, but variances are the same for all groups. Objects are assigned to group with nearest mean. -->


<!-- ## RDA with Grid Search -->

<!-- ```{r} -->
<!-- set.seed(1337) -->
<!-- cv_5_grid = trainControl(method = "cv", number = 5) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- set.seed(1337) -->
<!-- fit_rda_grid = train(Class ~ ., data = Sonar, method = "rda", trControl = cv_5_grid) -->
<!-- fit_rda_grid -->
<!-- ``` -->

<!-- ```{r} -->
<!-- plot(fit_rda_grid) -->
<!-- ``` -->

<!-- ## RDA with Random Search Search -->


<!-- ```{r} -->
<!-- set.seed(1337) -->
<!-- cv_5_rand = trainControl(method = "cv", number = 5, search = "random") -->
<!-- ``` -->

<!-- ```{r} -->
<!-- fit_rda_rand = train(Class ~ ., data = Sonar, method = "rda",  -->
<!--                      trControl = cv_5_rand, tuneLength = 9) -->
<!-- fit_rda_rand -->
<!-- ``` -->

<!-- ```{r} -->
<!-- ggplot(fit_rda_rand) -->
<!-- ``` -->


<!-- ## Comparison to Elastic Net -->

<!-- ```{r} -->
<!-- set.seed(1337) -->
<!-- fit_elnet_grid = train(Class ~ ., data = Sonar, method = "glmnet",  -->
<!--                        trControl = cv_5_grid, tuneLength = 10) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- set.seed(1337) -->
<!-- fit_elnet_int_grid = train(Class ~ . ^ 2, data = Sonar, method = "glmnet",  -->
<!--                            trControl = cv_5_grid, tuneLength = 10) -->
<!-- ``` -->


<!-- ## Results -->

<!-- ```{r} -->
<!-- get_best_result = function(caret_fit) { -->
<!--   best_result = caret_fit$results[as.numeric(rownames(caret_fit$bestTune)), ] -->
<!--   rownames(best_result) = NULL -->
<!--   best_result -->
<!-- } -->
<!-- ``` -->

<!-- ```{r} -->
<!-- knitr::kable(rbind( -->
<!--   get_best_result(fit_rda_grid), -->
<!--   get_best_result(fit_rda_rand))) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- knitr::kable(rbind( -->
<!--   get_best_result(fit_elnet_grid), -->
<!--   get_best_result(fit_elnet_int_grid))) -->

<!-- ``` -->


<!-- ## External Links -->

<!-- - [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) - Paper justifying random tuning parameter search. -->
<!-- - [Random Hyperparameter Search](https://topepo.github.io/caret/random-hyperparameter-search.html) - Details on random tuning parameter search in `caret`. -->


<!-- ## RMarkdown -->

<!-- The RMarkdown file for this chapter can be found [**here**](17-rda.Rmd). The file was created using `R` version 3.4.2 and the following packages: -->

<!-- - Base Packages, Attached -->

<!-- ```{r, echo = FALSE} -->
<!-- sessionInfo()$basePkgs -->
<!-- ``` -->

<!-- - Additional Packages, Attached -->

<!-- ```{r, echo = FALSE} -->
<!-- names(sessionInfo()$otherPkgs) -->
<!-- ``` -->

<!-- - Additional Packages, Not Attached -->

<!-- ```{r, echo = FALSE} -->
<!-- names(sessionInfo()$loadedOnly) -->
<!-- ``` -->

