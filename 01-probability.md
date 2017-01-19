---
output:
  html_document: default
  pdf_document: default
---
# Probability Review

We give a very brief review of some necessary probability concepts. As the treatment is less than complete, a list of references is given at the end of the chapter. For example, we ignore the usual recap of basic set theory and omit proofs and examples.


## Probability Models

When discussing probability models, we speak of random **experiments** that produce one of a number of possible **outcomes**.

A **probability model** that describes the uncertainty of an experiment consists of two elements:

- The **sample space**, often denoted as $\Omega$, which is a set which contains all possible outcomes.
-  A **probability function** that assigns to an event $A$ a nonnegative number, $P[A]$, that represents how likely it is that event $A$ occurs as a result of the experiment.

We call $P[A]$ the **probability** of event $A$. An **event** $A$ could be any subset of the sample space, not necessarily a single possible outcome. The probability law must follow a number of rules, which are the result of a set of axioms that we introduce now.


## Probability Axioms

Given a sample space $\Omega$ for a particular experiment, the **probability function** associated with the experiement must satisfy the following axioms.

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

If additionally $A$ and $B$ are *disjoint*, then we have:

$$
P[A \cup B] = P[A] + P[B]
$$

If we had $n$ mutually exclusive events, $E_1, E_2, \ldots E_n$, then we have:

$$
P\left[\textstyle\bigcup_{i = 1}^{n} E_i\right] = \sum_{i = 1}^{n} P[E_i]
$$

Often, we would like to understand the probability of an event $A$, given some information about the outcome of event $B$. In that case, we have the **conditional probability rule** provided $P[B] > 0$.  

$$
P[A \mid B] = \frac{P[A \cap B]}{P[B]}
$$

Rearranging the conditional probability rule, we obtain the **multiplication rule**:

$$
P[A \cap B] = P[A \mid B] \cdot P[B]
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

Two events $A$ and $B$ are said to be **independent** if they statisfy

$$
P[A \cap B] = P[A] \cdot P[B]
$$

This becomes the new multiplication rule for independent events.

A collection of events $E_1, E_2, \ldots E_n$ is said to be independent if

$$
P\left[\bigcup_{i \in S} E_i \right] = \prod_{i \in S}P[A_i]
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

- most commonly defined as possible values + pmf (probabilites)
- binomial as example,  parameters

### Continuous Random Variables

- most commonly defined as possible values + pdf, can also use cdf or mgf
- normal as example, parameters

### Several Random Variables

- discussion only two DRV, note extensions to several and continuous
- ind
- joint
- marginal
- conditional


## Expectations

$$
\mathbb{E}[f(X)] \triangleq \sum_{x} f(x)p(x)
$$

$$
\mathbb{E}[f(X)] \triangleq \int f(x)p(x) dx
$$

$$
\mu_{X} = \text{mean}[X] \triangleq \mathbb{E}[X]
$$

$$
\sigma^2_{X} = \text{var}[X] \triangleq \mathbb{E}[(X - \mathbb{E}[X])^2]
$$

$$
\sigma_{X} = \text{sd}[X] \triangleq \sqrt{\sigma^2_{X}} = \sqrt{\text{var}[X]}
$$

$$
\text{cov}[X, Y] \triangleq \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$


## Likelihood

- discussion of parameters

$$
\mathcal{L}(\theta \mid x_1, x_2, \ldots x_n)
$$


## Bayesian Nomenclature

- post
- prior
- likelihood
- post ~ prior * likelihood


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
    - [Introduction to Mathematical Statistics](https://www.pearsonhighered.com/program/Hogg-Introduction-to-Mathematical-Statistics-7th-Edition/PGM49624.html) by Robert V. Hogg, Joeseph McKean, and Allen T. Craig

