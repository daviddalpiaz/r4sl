# (PART) Regression {-}

# Overview {#regression-overview}

**TODO:** Take main concepts from the next chapter and talk about them in general here. The next chapter will illustrate them by example.

**Note:** This is a placeholder chapter that is new. 

- **Supervised Learning**
    - **Regression** (Numeric Response)
        - What do we want? To make *predictions* on *unseen data*. (Predicting on data we already have is easy...) In other words, we want a *model* that **generalizes** well. That is, generalizes to unseen data.
        - How we will do this? By controlling the **complexity** of the model to guard against **overfitting** and **underfitting**.
            - Model Parameters
            - Tuning Parameters
        - Why does manipulating the model complexity accomplish this? Because there is a **bias-variance tradeoff**.
        - How do we know if our model generalizes? By evaluating *metrics* on **test** data. We will only ever fit (train) models on training data. All analyses will begin with a test-train split. For regression tasks, our metric will be **RMSE**.
    - Classification (Categorical Response) The next section.

![](images/regression.png)

**Regression** is a form of **supervised learning**. Supervised learning deals with problems where there are both an input and an output. Regression problems are the subset of supervised learning problems with a **numeric** output.

Often one of the biggest differences between *statistical learning*, *machine learning*, *artificial intelligence* are the names used to describe variables and methods.

- The **input** can be called: input vector, feature vector, or predictors. The elements of these would be an input, feature, or predictor. The individual features can be either numeric or categorical.
- The **output** may be called: output, response, outcome, or target. The response must be numeric.

As an aside, some textbooks and statisticians use the terms independent and dependent variables to describe the response and the predictors. However, this practice can be confusing as those terms have specific meanings in probability theory.

*Our goal is to find a rule, algorithm, or function which takes as input a feature vector, and outputs a response which is as close to the true value as possible.* We often write the true, unknown relationship between the input and output $f(\bf{x})$. The relationship (model) we learn (fit, train), based on data, is written $\hat{f}(\bf{x})$.

From a statistical learning point-of-view, we write,

$$
Y = f(\bf{x}) + \epsilon
$$



to indicate that the true response is a function of both the unknown relationship, as well as some unlearnable noise.



$$
\text{RMSE}(\hat{f}, \text{Data}) = \sqrt{\frac{1}{n}\displaystyle\sum_{i = 1}^{n}\left(y_i - \hat{f}(\bf{x}_i)\right)^2}
$$

$$
\text{RMSE}_{\text{Train}} = \text{RMSE}(\hat{f}, \text{Train Data}) = \sqrt{\frac{1}{n_{\text{Tr}}}\displaystyle\sum_{i \in \text{Train}}^{}\left(y_i - \hat{f}(\bf{x}_i)\right)^2}
$$

$$
\text{RMSE}_{\text{Test}} = \text{RMSE}(\hat{f}, \text{Test Data}) = \sqrt{\frac{1}{n_{\text{Te}}}\displaystyle\sum_{i \in \text{Test}}^{}\left(y_i - \hat{f}(\bf{x}_i)\right)^2}
$$
