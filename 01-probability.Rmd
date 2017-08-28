# Probability Review

We give a very brief review of some necessary probability concepts. As the treatment is less than complete, a list of references is given at the end of the chapter. For example, we ignore the usual recap of basic set theory and omit proofs and examples.


## Probability Models

When discussing probability models, we speak of random **experiments** that produce one of a number of possible **outcomes**.

A **probability model** that describes the uncertainty of an experiment consists of two elements:

- The **sample space**, often denoted as $\Omega$, which is a set that contains all possible outcomes.
-  A **probability function** that assigns to an event $A$ a nonnegative number, $P[A]$, that represents how likely it is that event $A$ occurs as a result of the experiment.

We call $P[A]$ the **probability** of event $A$. An **event** $A$ could be any subset of the sample space, not necessarily a single possible outcome. The probability law must follow a number of rules, which are the result of a set of axioms that we introduce now.


## Probability Axioms

Given a sample space $\Omega$ for a particular experiment, the **probability function** associated with the experiment must satisfy the following axioms.

1. *Nonnegativity*: $P[A] \geq 0$ for any event $A \subset \Omega$.
2. *Normalization*: $P[\Omega] = 1$. That is, the probability of the entire space is 1.
3. *Additivity*: For mutually exclusive events $E_1, E_2, \ldots$
$$
P\left[\bigcup_{i = 1}^{\infty} E_i\right] = \sum_{i = 1}^{\infty} P[E_i]
$$

Using these axioms, many additional probability rules can easily be derived.


## Probability Rules

Given an event $A$, and its complement, $A^c$, that is, the outcomes in $\Omega$ which are not in $A$, we have the **complement rule**:

$$
P[A^c] = 1 - P[A]
$$

In general, for two events $A$ and $B$, we have the **addition rule**:

$$
P[A \cup B] = P[A] + P[B] - P[A \cap B]
$$

If $A$ and $B$ are also *disjoint*, then we have:

$$
P[A \cup B] = P[A] + P[B]
$$

If we have $n$ mutually exclusive events, $E_1, E_2, \ldots E_n$, then we have:

$$
P\left[\textstyle\bigcup_{i = 1}^{n} E_i\right] = \sum_{i = 1}^{n} P[E_i]
$$

Often, we would like to understand the probability of an event $A$, given some information about the outcome of event $B$. In that case, we have the **conditional probability rule** provided $P[B] > 0$.  

$$
P[A \mid B] = \frac{P[A \cap B]}{P[B]}
$$

Rearranging the conditional probability rule, we obtain the **multiplication rule**:

$$
P[A \cap B] = P[B] \cdot P[A \mid B] \cdot 
$$

For a number of events $E_1, E_2, \ldots E_n$, the multiplication rule can be expanded into the **chain rule**:

$$
P\left[\textstyle\bigcap_{i = 1}^{n} E_i\right] = P[E_1] \cdot P[E_2 \mid E_1] \cdot P[E_3 \mid E_1 \cap E_2] \cdots P\left[E_n \mid \textstyle\bigcap_{i = 1}^{n - 1} E_i\right] 
$$

Define a **partition** of a sample space $\Omega$ to be a set of disjoint events $A_1, A_2, \ldots, A_n$ whose union is the sample space $\Omega$. That is

$$
A_i \cap A_j = \emptyset
$$

for all $i \neq j$, and

$$
\bigcup_{i = 1}^{n} A_i = \Omega.
$$

Now, let $A_1, A_2, \ldots, A_n$ form a partition of the sample space where $P[A_i] > 0$ for all $i$. Then for any event $B$ with $P[B] > 0$ we have **Bayes' Rule**:

$$
P[A_i | B] = \frac{P[A_i]P[B | A_i]}{P[B]} = \frac{P[A_i]P[B | A_i]}{\sum_{i = 1}^{n}P[A_i]P[B | A_i]}
$$

The denominator of the latter equality is often called the **law of total probability**:

$$
P[B] = \sum_{i = 1}^{n}P[A_i]P[B | A_i]
$$

Two events $A$ and $B$ are said to be **independent** if they satisfy

$$
P[A \cap B] = P[A] \cdot P[B]
$$

This becomes the new multiplication rule for independent events.

A collection of events $E_1, E_2, \ldots E_n$ is said to be independent if

$$
P\left[\bigcap_{i \in S} E_i \right] = \prod_{i \in S}P[A_i]
$$

for every subset $S$ of $\{1, 2, \ldots n\}$.

If this is the case, then the chain rule is greatly simplified to:

$$
P\left[\textstyle\bigcap_{i = 1}^{n} E_i\right] = \prod_{i=1}^{n}P[A_i]
$$


## Random Variables

A **random variable** is simply a *function* which maps outcomes in the sample space to real numbers.


### Distributions

We often talk about the **distribution** of a random variable, which can be thought of as:

$$
\text{distribution} = \text{list of possible} \textbf{ values} + \text{associated} \textbf{ probabilities}
$$

This is not a strict mathematical definition, but is useful for conveying the idea.

If the possible values of a random variables are *discrete*, it is called a *discrete random variable*. If the possible values of a random variables are *continuous*, it is called a *continuous random variable*. 


### Discrete Random Variables

The distribution of a discrete random variable $X$ is most often specified by a list of possible values and a probability **mass** function, $p(x)$. The mass function directly gives probabilities, that is, 

$$
p(x) = p_X(x) = P[X = x].
$$

Note we almost always drop the subscript from the more correct $p_X(x)$ and simply refer to $p(x)$. The relevant random variable is discerned from context

The most common example of a discrete random variable is a **binomial** random variable. The mass function of a binomial random variable $X$, is given by

$$
p(x | n, p) = {n \choose x} p^x(1 - p)^{n - x}, \ \ \ x = 0, 1, \ldots, n, \ n \in \mathbb{N}, \ 0 < p < 1.
$$

This line conveys a large ammount of information.

- The function $p(x | n, p)$ is the mass function. It is a function of $x$, the possible values of the random variable $X$. It is conditional on the **parameters** $n$ and $p$. Different values of these parameters specify different binomial distributions.
- $x = 0, 1, \ldots, n$ indicates the **sample space**, that is, the possible values of the random variable.
- $n \in \mathbb{N}$ and $0 < p < 1$ specify the **parameter spaces**. These are the possible values of the parameters that give a valid binomial distribution.

Often all of this information is simply encoded by writing

$$
X \sim \text{bin}(n, p).
$$


### Continuous Random Variables

The distribution of a continuous random variable $X$ is most often specified by a set of possible values and a probability **density** function, $f(x)$. (A cumulative density or moment generating function would also suffice.)

The probability of the event $a < X < b$ is calculated as

$$
P[a < X < b] = \int_{a}^{b} f(x)dx.
$$

Note that densities are **not** probabilities.

The most common example of a continuous random variable is a **normal** random variable. The density of a normal random variable $X$, is given by 

$$
f(x | \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \cdot \exp\left[\frac{-1}{2} \left(\frac{x - \mu}{\sigma}\right)^2 \right],  \ \ \ -\infty < x < \infty, \ -\infty < \mu < \infty, \ \sigma > 0.
$$

- The function $f(x | \mu, \sigma^2)$ is the density function. It is a function of $x$, the possible values of the random variable $X$. It is conditional on the **paramters** $\mu$ and $\sigma^2$. Different values of these parameters specify different normal distributions.
- $-\infty < x < \infty$ indicates the sample space. In this case, the random variable may take any value on the real line.
- $-\infty < \mu < \infty$ and $\sigma > 0$ specify the parameter space. These are the possible values of the parameters that give a valid normal distribution.

Often all of this information is simply encoded by writing

$$
X \sim N(\mu, \sigma^2)
$$

### Several Random Variables

Consider two random variables $X$ and $Y$. We say they are independent if

$$
f(x, y) = f(x) \cdot f(y)
$$

for all $x$ and $y$. Here $f(x, y)$ is the **joint** density (mass) function of $X$ and $Y$. We call $f(x)$ the **marginal** density (mass) function of $X$. Then $f(y)$ the marginal density (mass) function of $Y$. The joint density (mass) function $f(x, y)$ together with the possible $(x, y)$ values specify the joint distribution of $X$ and $Y$.

Similar notions exist for more than two variables.


## Expectations

For discrete random variables, we define the **expectation** of the function of a random variable $X$ as follows.

$$
\mathbb{E}[g(X)] \triangleq \sum_{x} g(x)p(x)
$$

For continuous random variables we have a similar definition.

$$
\mathbb{E}[g(X)] \triangleq \int g(x)f(x) dx
$$

For specific functions $g$, expectations are given names.

The **mean** of a random variable $X$ is given by

$$
\mu_{X} = \text{mean}[X] \triangleq \mathbb{E}[X].
$$

So for a discrete random variable, we would have

$$
\text{mean}[X] = \sum_{x} x \cdot p(x)
$$

For a continuous random variable we would simply replace the sum by an integral.

The **variance** of a random variable $X$ is given by

$$
\sigma^2_{X} = \text{var}[X] \triangleq \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.
$$

The **standard deviation of a random variable $X$ is given by

$$
\sigma_{X} = \text{sd}[X] \triangleq \sqrt{\sigma^2_{X}} = \sqrt{\text{var}[X]}.
$$

The **covariance** or random variables $X$ and $Y$ is given by

$$
\text{cov}[X, Y] \triangleq \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X] \cdot \mathbb{E}[Y].
$$


## Likelihood

Consider $n$ iid random variables $X_1, X_2, \ldots X_n$. We can then write their **likelihood** as

$$
\mathcal{L}(\theta \mid x_1, x_2, \ldots x_n) = \prod_{i = i}^n f(x_i; \theta)
$$

where $f(x_i; \theta)$ is the density (or mass) function of random variable $X_i$ evaluated at $x_i$ with parameter $\theta$.

Whereas a probability is a function of a possible observed value given a particular parameter value, a likelihood is the opposite. It is a function of a possible parameter value given observed data.

Maximumizing likelihood is a common techinque for fitting a model to data.

## Videos

The YouTube channel [mathematicalmonk](https://www.youtube.com/user/mathematicalmonk) has a great [Probability Primer playlist](https://www.youtube.com/playlist?list=PL17567A1A3F5DB5E4) containing lectures on many fundamental probability concepts. Some of the more important concepts are covered in the following videos:

- [Conditional Probability](https://www.youtube.com/watch?v=5BWk5qe5EJ8&index=11&list=PL17567A1A3F5DB5E4)
- [Independence](https://www.youtube.com/watch?v=KK9jvGl9FY0&index=12&list=PL17567A1A3F5DB5E4)
- [More Independence](https://www.youtube.com/watch?v=RMS-WglZP-c&index=13&list=PL17567A1A3F5DB5E4)
- [Bayes Rule](https://www.youtube.com/watch?v=cM1BqBv11U8&index=14&list=PL17567A1A3F5DB5E4)

## References

Any of the following are either dedicated to, or contain a good coverage of the details of the topics above.

- Probability Texts
    - [Introduction to Probability](http://athenasc.com/probbook.html) by Dimitri P. Bertsekas and John N. Tsitsiklis
    - [A First Course in Probability](https://www.pearsonhighered.com/program/Ross-First-Course-in-Probability-A-9th-Edition/PGM110742.html) by Sheldon Ross
- Machine Learning Texts with Probability Focus
    - [Probability for Statistics and Machine Learning](http://www.springer.com/us/book/9781441996336) by Anirban DasGupta
    - [Machine Learning: A Probabilistic Perspective](https://mitpress.mit.edu/books/machine-learning-0) by Kevin P. Murphy
- Statistics Texts with Introduction to Probability
    - [Probability and Statistical Inference](https://www.pearsonhighered.com/program/Hogg-Probability-and-Statistical-Inference-9th-Edition/PGM91556.html) by Robert V. Hogg, Elliot Tanis, and Dale Zimmerman
    - [Introduction to Mathematical Statistics](https://www.pearsonhighered.com/program/Hogg-Introduction-to-Mathematical-Statistics-7th-Edition/PGM49624.html) by Robert V. Hogg, Joseph McKean, and Allen T. Craig

