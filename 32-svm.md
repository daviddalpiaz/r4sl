# Support Vector Machines

**TOOD:** This chpater is currently empty to reduce build time.

<!-- ## `R` Packages -->

<!-- ```{r, message = FALSE, warning = FALSE} -->
<!-- library(randomForest) -->
<!-- library(caret) -->
<!-- library(kernlab) -->
<!-- ``` -->

<!-- For working with SVMs in `R` we will use the [kernlab package](https://cran.r-project.org/web/packages/kernlab/vignettes/kernlab.pdf) instead of `e1071`. For examples of `e1071` see the relevant chapter in ISL. We do not discuss the details of the method, however show how the method can be tuned. -->

<!-- ## Classification -->

<!-- SVMs are mostly used for classification. Note that they can be modified for regression but we will not do that here. Since we are performing classification, we will use accuracy as our metric. -->

<!-- ```{r} -->
<!-- accuracy = function(actual, predicted) { -->
<!--   mean(actual == predicted) -->
<!-- } -->
<!-- ``` -->

<!-- ## Linear, Separable Example -->

<!-- ### Data Simulation -->

<!-- ```{r} -->
<!-- sim_sep = function(n = 1000) { -->
<!--   x1 = runif(n) -->
<!--   x2 = runif(n) -->
<!--   keep = x1 + 0.1 < x2 | x1 - 0.1 > x2 -->
<!--   x1 = x1[keep] -->
<!--   x2 = x2[keep] -->
<!--   y = 1 * (x1 - x2 > 0) -->
<!--   y = ifelse(y == 1, "Orange", "Blue") -->
<!--   data.frame(y = as.factor(y), x1 = x1, x2 = x2) -->
<!-- } -->
<!-- ``` -->

<!-- ```{r} -->
<!-- set.seed(42) -->
<!-- train_data = sim_sep(n = 50) -->
<!-- plot(x2 ~ x1, data = train_data, col = as.character(y), pch = 19) -->
<!-- test_data = sim_sep(n = 500) -->
<!-- str(train_data) -->
<!-- ``` -->

<!-- ### Linear Kernel, Parameter `C` -->

<!-- ```{r} -->
<!-- lin_svm_fit = ksvm(y ~ ., data = train_data, kernel = 'vanilladot', C = 0.1) -->
<!-- plot(lin_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(lin_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(lin_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- lin_svm_fit = ksvm(y ~ ., data = train_data, kernel = 'vanilladot', C = 1) -->
<!-- plot(lin_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(lin_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(lin_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- lin_svm_fit = ksvm(y ~ ., data = train_data, kernel = 'vanilladot', C = 10) -->
<!-- plot(lin_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(lin_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(lin_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Radial Kernel -->

<!-- ```{r} -->
<!-- set.seed(42) -->
<!-- rad_svm_fit = ksvm(y ~ ., data = train_data, kernel = 'rbfdot', C = 1) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Tuning with `caret` -->

<!-- ```{r} -->
<!-- svm_grid =  expand.grid(C = c(2 ^ (-5:5))) -->
<!-- svm_control = trainControl(method = "cv", number = 5, -->
<!--                            returnResamp = "all", verbose = FALSE) -->

<!-- set.seed(42) -->
<!-- lin_svm_fit = train(y ~ ., data = train_data, method = "svmLinear", -->
<!--                     trControl = svm_control, tuneGrid = svm_grid) -->

<!-- lin_svm_fit -->
<!-- lin_svm_fit$bestTune -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(lin_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(lin_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Compare: Random Forest -->

<!-- ```{r} -->
<!-- set.seed(42) -->
<!-- rf_grid = expand.grid(mtry = 1:2) -->
<!-- rf_fit  = train(y ~ ., data = train_data, method = "rf", -->
<!--                 trControl = svm_control, tuneGrid = rf_grid) -->
<!-- rf_fit$bestTune -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rf_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rf_fit, test_data)) -->
<!-- ``` -->

<!-- ## Non-Linear, Non-Separable Example -->

<!-- ### Data Simulation -->

<!-- ```{r} -->
<!-- sim_rad = function(n = 1000) { -->
<!--   x1 = runif(n, -1, 1) -->
<!--   x2 = runif(n, -1, 1) -->
<!--   y = 1 * (x1 ^ 2 + x2 ^ 2 + rnorm(n, 0, 0.25) < 0.5) -->
<!--   y = ifelse(y == 1, "Orange", "Blue") -->
<!--   data.frame(y = as.factor(y), x1 = x1, x2 = x2) -->
<!-- } -->
<!-- ``` -->

<!-- ```{r} -->
<!-- set.seed(42) -->
<!-- train_data = sim_rad(n = 250) -->
<!-- plot(x2 ~ x1, data = train_data, col = as.character(y), pch = 19) -->
<!-- test_data = sim_rad(n = 2000) -->
<!-- ``` -->

<!-- ### Radial Kernel, Parameter `C` -->

<!-- ```{r} -->
<!-- rad_svm_fit = ksvm(y ~., data = train_data, kernel = 'rbfdot', -->
<!--                    C = 0.1, kpar = list(sigma = 1)) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- rad_svm_fit = ksvm(y ~., data = train_data, kernel = 'rbfdot', -->
<!--                    C = 1, kpar = list(sigma = 1)) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->


<!-- ```{r} -->
<!-- rad_svm_fit = ksvm(y ~., data = train_data, kernel = 'rbfdot', -->
<!--                    C = 10, kpar = list(sigma = 1)) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Radial Kernel, Parameter `sigma` -->

<!-- ```{r} -->
<!-- rad_svm_fit = ksvm(y ~., data = train_data, kernel = 'rbfdot', -->
<!--                    C = 1, kpar = list(sigma = 0.5)) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- rad_svm_fit = ksvm(y ~., data = train_data, kernel = 'rbfdot', -->
<!--                    C = 1, kpar = list(sigma = 1)) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ```{r} -->
<!-- rad_svm_fit = ksvm(y ~., data = train_data, kernel = 'rbfdot', -->
<!--                    C = 1, kpar = list(sigma = 2)) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Radial Kernel, Tuning -->

<!-- ```{r} -->
<!-- svm_grid = expand.grid(   C = c(2 ^ (-5:5)), -->
<!--                       sigma = c(2 ^ (-3:3))) -->

<!-- set.seed(42) -->
<!-- rad_svm_fit = train(y ~ ., data = train_data, method = "svmRadial", -->
<!--                     trControl = svm_control, tuneGrid = svm_grid) -->
<!-- #rad_svm_fit -->
<!-- rad_svm_fit$bestTune -->
<!-- ``` -->

<!-- ```{r} -->
<!-- rad_svm_fit = ksvm(y ~., data = train_data, kernel = 'rbfdot', -->
<!--                   C = 16, kpar = list(sigma = 0.25)) -->
<!-- plot(rad_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rad_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rad_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Polynomial Kernel, Tuning -->

<!-- ```{r} -->
<!-- set.seed(42) -->
<!-- poly_svm_fit = train(y ~ ., data = train_data, method = "svmPoly", -->
<!--                      trControl = svm_control) -->
<!-- #poly_svm_fit -->
<!-- poly_svm_fit$bestTune -->
<!-- ``` -->

<!-- ```{r} -->
<!-- poly_svm_fit = ksvm(y ~., data = train_data, kernel = 'polydot', -->
<!--                     C = 1, kpar = list(scale = 0.1, degree = 3)) -->
<!-- plot(poly_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(poly_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(poly_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Linear Kernel, Tuning -->

<!-- ```{r} -->
<!-- svm_grid =  expand.grid(C = c(2 ^ (-5:5))) -->
<!-- set.seed(42) -->
<!-- lin_svm_fit = train(y ~ ., data = train_data, method = "svmLinear", -->
<!--                     trControl = svm_control, tuneGrid = svm_grid) -->
<!-- lin_svm_fit -->
<!-- lin_svm_fit$bestTune -->
<!-- ``` -->

<!-- ```{r} -->
<!-- lin_svm_fit = ksvm(y ~., data = train_data, kernel = 'vanilladot', -->
<!--                    C = lin_svm_fit$bestTune) -->
<!-- plot(lin_svm_fit, data = train_data) -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(lin_svm_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(lin_svm_fit, test_data)) -->
<!-- ``` -->

<!-- ### Compare: Random Forest -->

<!-- ```{r} -->
<!-- set.seed(42) -->
<!-- rf_grid = expand.grid(mtry = 1:2) -->
<!-- rf_fit = train(y ~ ., data = train_data, method = "rf", -->
<!--                trControl = svm_control, tuneGrid = rf_grid) -->
<!-- rf_fit$bestTune -->

<!-- # train accuracy -->
<!-- accuracy(actual = train_data$y, -->
<!--          predicted = predict(rf_fit, train_data)) -->

<!-- # test accuracy -->
<!-- accuracy(actual = test_data$y, -->
<!--          predicted = predict(rf_fit, test_data)) -->
<!-- ``` -->



<!-- ## External Links -->

<!-- - [SVM with Polynomial Kernel Visualization](https://www.youtube.com/watch?v=3liCbRZPrZA) - The kernel idea in one simple video. -->


<!-- ## RMarkdown -->

<!-- The RMarkdown file for this chapter can be found [**here**](21-svm.Rmd). The file was created using `R` version 3.5.2 and the following packages: -->

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



