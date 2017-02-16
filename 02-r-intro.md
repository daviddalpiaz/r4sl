# Introduction to `R`

> "Measuring programming progress by lines of code is like measuring aircraft building progress by weight."
>
> --- **Bill Gates**

After reading this chapter you will be able to:

- Interact with `R` using RStudio.
- Use `R` as a calculator.
- Work with data as vectors and data frames.
- Make basic data visualizations.
- Write your own `R` functions.
- Perform hypothesis tests using `R`.
- Perform basic simulations in `R`.

## `R` Resources

`R` is both a programming language and software environment for statistical computing, which is *free* and *open-source*. To get started, you will need to install two pieces of software:

- [`R`, the actual programming language.](http://cran.r-project.org/)
    - Chose your operating system, and select the most recent version, 3.3.2.
- [RStudio, an excellent IDE for working with `R`.](http://www.rstudio.com/)
    - Note, you must have `R` installed to use RStudio. RStudio is simply an interface used to interact with `R`.

The popularity of `R` is on the rise, and everyday it becomes a better tool for statistical analysis. It even generated this book! (A skill you will learn in this course.) There are many good resources for learning `R`. They are not necessary for this course, but you may find them useful if you would like a deeper understanding of `R`:

- [Try R](http://tryr.codeschool.com/) from Code School.
    - An interactive introduction to the basics of `R`. Could be very useful for getting up to speed on `R`'s syntax.
- [Quick-R](http://www.statmethods.net/) by Robert Kabacoff.
    - A good reference for `R` basics.
- [R Tutorial](http://www.r-tutor.com/) by Chi Yau.
    - A combination reference and tutorial for `R` basics.
- [R Markdown](http://rmarkdown.rstudio.com/) from RStudio.
    - Reference materials for RMarkdown.
- [The Art of `R` Programming](https://www.nostarch.com/artofr.htm) by Norman Matloff.
    - Gentle introduction to the programming side of `R`. (Whereas we will focus more on the data analysis side.) A [free electronic version](http://vufind.carli.illinois.edu/vf-uiu/Record/uiu_6955421) is available through the Illinois library.
- [Advanced `R`](http://adv-r.had.co.nz/) by Hadley Wickham.
    - From the author of several extremely popular `R` packages. Good follow-up to The Art of `R` Programming. (And more up-to-date material.)
- [`R` for Data Science](http://r4ds.had.co.nz/) by Hadley Wickham and Garrett Grolemund.
    - Similar to Advanced `R`, but focuses more on data analysis, while still introducing programming concepts. At the time of writing, currently under development.
- [The R Inferno](http://www.burns-stat.com/documents/books/the-r-inferno/) by Patrick Burns.
    - Likens learning the tricks of `R` to descending through the levels of hell. Very advanced material, but may be important if `R` becomes a part of your everyday toolkit.

RStudio has a large number of useful keyboard shortcuts. A list of these can be found using a keyboard shortcut -- the keyboard shortcut to rule them all:

- On Windows: `Alt` + `Shift` + `K`
- On Mac:  `Option` + `Shift` + `K`

The RStudio team has developed [a number of "cheatsheets"](https://www.rstudio.com/resources/cheatsheets/) for working with both `R` and RStudio. [This particular cheatseet for Base `R`](http://www.rstudio.com/wp-content/uploads/2016/05/base-r.pdf) will summarize many of the concepts in this document.

When programming, it is often a good practice to follow a style guide. (Where do spaces go? Tabs or spaces? Underscores or CamelCase when naming variables?) No style guide is "correct" but it helps to be aware of what others do. The more import thing is to be consistent within your own code.

- [Hadley Wickham Style Guide](http://adv-r.had.co.nz/Style.html) from [Advanced `R`](http://adv-r.had.co.nz/)
- [Google Style Guide](https://google.github.io/styleguide/Rguide.xml)

For this course, our main deviation from these two guides is the use of `=` in place of `<-`. (More on that later.)

## `R` Basics

### Basic Calculations

To get started, we'll use `R` like a simple calculator.

#### Addition, Subtraction, Multiplication and Division {-}

| Math          | `R`     | Result    |
|---------------|---------|-----------|
| $3 + 2$       | `3 + 2` | 5 |
| $3 - 2$       | `3 - 2` | 1 |
| $3 \cdot2$    | `3 * 2` | 6 |
| $3 / 2$       | `3 / 2` | 1.5 |

#### Exponents  {-}

| Math         | `R`             | Result            |
|--------------|-----------------|-------------------|
| $3^2$        | `3 ^ 2`         | 9         |
| $2^{(-3)}$   | `2 ^ (-3)`      | 0.125      |
| $100^{1/2}$  | `100 ^ (1 / 2)` | 10 |
| $\sqrt{100}$ | `sqrt(100)`     | 10     |

#### Mathematical Constants  {-}

| Math         | `R`             | Result            |
|--------------|-----------------|-------------------|
| $\pi$        | `pi`            | 3.1415927            |
| $e$          | `exp(1)`        | 2.7182818        |

#### Logarithms  {-}

Note that we will use $\ln$ and $\log$ interchangeably to mean the natural logarithm. There is no `ln()` in `R`, instead it uses `log()` to mean the natural logarithm.

| Math              | `R`                 | Result                |
|-------------------|---------------------|-----------------------|
| $\log(e)$         | `log(exp(1))`       | 1       |
| $\log_{10}(1000)$ | `log10(1000)`       | 3       |
| $\log_{2}(8)$     | `log2(8)`           | 3           |
| $\log_{4}(16)$    | `log(16, base = 4)` | 2 |

#### Trigonometry  {-}

| Math            | `R`           | Result          |
|-----------------|---------------|-----------------|
| $\sin(\pi / 2)$ | `sin(pi / 2)` | 1 |
| $\cos(0)$       | `cos(0)`      | 1      |

### Getting Help

In using `R` as a calculator, we have seen a number of functions: `sqrt()`, `exp()`, `log()` and `sin()`. To get documentation about a function in `R`, simply put a question mark in front of the function name and RStudio will display the documentation, for example: 


```r
?log
?sin
?paste
?lm
```

Frequently one of the most difficult things to do when learning `R` is asking for help. First, you need to decide to ask for help, then you need to know *how* to ask for help. Your very first line of defense should be to Google your error message or a short description of your issue. (The ability to solve problems using this method is quickly becoming an extremely valuable skill.) If that fails, and it eventually will, you should ask for help. There are a number of things you should include when emailing an instructor, or posting to a help website such as [Stack Exchange](http://stats.stackexchange.com/).

- Describe what you expect the code to do.
- State the end goal you are trying to achieve. (Sometimes what you expect the code to do, is not what you want to actually do.)
- Provide the full text of any errors you have received.
- Provide enough code to recreate the error. Often for the purpose of this course, you could simply email your entire `.R` or `.Rmd` file.
- Sometimes it is also helpful to include a screenshot of your entire RStudio window when the error occurs.

If you follow these steps, you will get your issue resolved much quicker, and possibly learn more in the process. Do not be discouraged by running into errors and difficulties when learning `R`. (Or any technical skill.) It is simply part of the learning process.

### Installing Packages

`R` comes with a number of built-in functions and datasets, but one of the main strengths of `R` as an open-source project is its package system. Packages add additional functions and data. Frequently if you want to do something in `R`, and it isn't available by default, there is a good chance that there is a package that will fulfill your needs.

To install a package, use the `install.packages()` function. Think of this as buying a recipe book from the store, bringing it home, and putting it on your shelf.


```r
install.packages("ggplot2")
```

Once a package is installed, it must be loaded into your current `R` session before being used. Think of this as taking the book off of the shelf and opening it up to read.


```r
library(ggplot2)
```

Once you close `R`, all the packages are closed and put back on the imaginary shelf. The next time you open `R`, you do not have to install the package again, but you do have to load any packages you intend to use by invoking `library()`.


### Data Types

`R` has a number of basic data *types*.

- Numeric
    - Also known as Double. The default type when dealing with numbers.
    - Examples: `1`, `1.0`, `42.5`
- Integer
    - Examples: `1L`, `2L`, `42L`
- Complex
    - Example: `4 + 2i`
- Logical
    - Two possible values: `TRUE` and `FALSE`
    - You can also use `T` and `F`, but this is *not* recommended.
    - `NA` is also considered logical.
- Character
    - Examples: `"a"`, `"Statistics"`, `"1 plus 2."`

`R` also has a number of basic data *structures*. A data structure is either homogeneous (all elements are of the same data type) or heterogeneous (elements can be of more than one data type).

| Dimension | **Homogeneous** | **Heterogeneous** |
|-----------|-----------------|-------------------|
| 1         | Vector          | List              |
| 2         | Matrix          | Data Frame        |
| 3+        | Array           |                   |

### Vectors

Many operations in `R` make heavy use of **vectors**. Vectors in `R` are indexed starting at `1`. That is what the `[1]` in the output is indicating, that the first element of the row being displayed is the first element of the vector. Larger vectors will start additional rows with `[*]` where `*` is the index of the first element of the row.

Possibly the most common way to create a vector in `R` is using the `c()` function, which is short for "combine."" As the name suggests, it combines a list of numbers separated by commas. 


```r
c(1, 3, 5, 7, 8, 9)
```

```
## [1] 1 3 5 7 8 9
```

Here `R` simply outputs this vector. If we would like to store this vector in a **variable** we can do so with the **assignment** operator `=`. In this case the variable `x` now holds the vector we just created, and we can access the vector by typing `x`.


```r
x = c(1, 3, 5, 7, 8, 9)
x
```

```
## [1] 1 3 5 7 8 9
```

As an aside, there is a long history of the assignment operator in `R`, partially due to the keys available on the [keyboards of the creators of the `S` language.](https://twitter.com/kwbroman/status/747829864091127809) (Which preceded `R`.) For simplicity we will use `=`, but know that often you will see `<-` as the assignment operator. 

The pros and cons of these two are well beyond the scope of this book, but know that for our purposes you will have no issue if you simply use `=`. If you are interested in the weird cases where the difference matters, check out [The R Inferno](http://www.burns-stat.com/documents/books/the-r-inferno/).

If you wish to use `<-`, you will still need to use `=`, however only for argument passing. Some users like to keep assignment (`<-`) and argument passing (`=`) separate. No matter what you choose, the more important thing is that you **stay consistent**. Also, if working on a larger collaborative project, you should use whatever style is already in place.

Frequently you may wish to create a vector based on a sequence of numbers. The quickest and easiest way to do this is with the `:` operator, which creates a sequence of integers between two specified integers.


```r
(y = 1:100)
```

```
##   [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
##  [18]  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34
##  [35]  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51
##  [52]  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68
##  [69]  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85
##  [86]  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100
```

Here we see `R` labeling the rows after the first since this is a large vector. Also, we see that by putting parentheses around the assignment, `R` both stores the vector in a variable called `y` and automatically outputs `y` to the console.

To subset a vector, we use square brackets, `[]`. 


```r
x
```

```
## [1] 1 3 5 7 8 9
```

```r
x[1]
```

```
## [1] 1
```

```r
x[3]
```

```
## [1] 5
```

We see that `x[1]` returns the first element, and `x[3]` returns the third element.


```r
x[-2]
```

```
## [1] 1 5 7 8 9
```

We can also exclude certain indexes, in this case the second element.


```r
x[1:3]
```

```
## [1] 1 3 5
```

```r
x[c(1,3,4)]
```

```
## [1] 1 5 7
```

Lastly we see that we can subset based on a vector of indices.

One of the biggest strengths of `R` is its use of vectorized operations. (Frequently the lack of understanding of this concept leads of a belief that `R` is *slow*. `R` is not the fastest language, but it has a reputation for being slower than it really is.)


```r
x = 1:10
x + 1
```

```
##  [1]  2  3  4  5  6  7  8  9 10 11
```

```r
2 * x
```

```
##  [1]  2  4  6  8 10 12 14 16 18 20
```

```r
2 ^ x
```

```
##  [1]    2    4    8   16   32   64  128  256  512 1024
```

```r
sqrt(x)
```

```
##  [1] 1.000000 1.414214 1.732051 2.000000 2.236068 2.449490 2.645751
##  [8] 2.828427 3.000000 3.162278
```

```r
log(x)
```

```
##  [1] 0.0000000 0.6931472 1.0986123 1.3862944 1.6094379 1.7917595 1.9459101
##  [8] 2.0794415 2.1972246 2.3025851
```

We see that when a function like `log()` is called on a vector `x`, a vector is returned which has applied the function to each element of the vector  `x`.


```r
vec_1 = 1:10
vec_2 = 1:1000
vec_3 = 42
```

The length of a vector can be obtained with the `length()` function.


```r
length(vec_1)
```

```
## [1] 10
```

```r
length(vec_2)
```

```
## [1] 1000
```

```r
length(vec_3)
```

```
## [1] 1
```

Note that scalars do not exists in `R`. They are simply vectors of length `1`.

If we want to create a sequence that isn't limited to integers and increasing by 1 at a time, we can use the `seq()` function.


```r
seq(from = 1.5, to = 4.2, by = 0.1)
```

```
##  [1] 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1
## [18] 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2
```

We will discuss functions in detail later, but note here that the input labels `from`, `to`, and `by` are optional.


```r
seq(1.5, 4.2, 0.1)
```

```
##  [1] 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1
## [18] 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2
```

Another common operation to create a vector is `rep()`, which can repeat a single value a number of times.


```r
rep("A", times = 10)
```

```
##  [1] "A" "A" "A" "A" "A" "A" "A" "A" "A" "A"
```

Or, `rep()` can be used to repeat a vector a number of times.


```r
rep(x, times = 3)
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10  1  2  3  4  5  6  7  8  9 10  1  2  3
## [24]  4  5  6  7  8  9 10
```

We have now seen four different ways to create vectors:

- `c()`
- `:`
- `seq()`
- `rep()`

So far we have mostly used them in isolation, but they are often used together.


```r
c(x, rep(seq(1, 9, 2), 3), c(1, 2, 3), 42, 2:4)
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10  1  3  5  7  9  1  3  5  7  9  1  3  5
## [24]  7  9  1  2  3 42  2  3  4
```

### Summary Statistics

`R` has built in functions for a large number of summary statistics.


```r
y
```

```
##   [1]   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
##  [18]  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34
##  [35]  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51
##  [52]  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68
##  [69]  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85
##  [86]  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100
```

#### Central Tendency {-}

| Measure | `R`         | Result        |
|---------|-------------|---------------|
| Mean    | `mean(y)`   | 50.5   |
| Median  | `median(y)` | 50.5 |

#### Spread {-}

| Measure            | `R`        | Result       |
|--------------------|------------|--------------|
| Variance           | `var(y)`   | 841.6666667   |
| Standard Deviation | `sd(y)`    | 29.011492    |
| IQR                | `IQR(y)`   | 49.5   |
| Minimum            | `min(y)`   | 1   |
| Maximum            | `max(y)`   | 100   |
| Range              | `range(y)` | 1, 100 |


### Matrices

`R` can also be used for **matrix** calculations. Matrices have rows and columns containing a single data type. In a matrix, the order of rows and columns is important. (This is not true of *data frames*, which we will see later.)

Matrices can be created using the `matrix` function. 


```r
x = 1:9
x
```

```
## [1] 1 2 3 4 5 6 7 8 9
```

```r
X = matrix(x, nrow = 3, ncol = 3)
X
```

```
##      [,1] [,2] [,3]
## [1,]    1    4    7
## [2,]    2    5    8
## [3,]    3    6    9
```

Note here that we are using two different variables: lower case `x`, which stores a vector and capital `X`, which stores a matrix. (Following the usual mathematical convention.) We can do this because `R` is case sensitive.

By default the `matrix` function reorders a vector into columns, but we can also tell `R` to use rows instead.


```r
Y = matrix(x, nrow = 3, ncol = 3, byrow = TRUE)
Y
```

```
##      [,1] [,2] [,3]
## [1,]    1    2    3
## [2,]    4    5    6
## [3,]    7    8    9
```

We can also create a matrix of a specified dimension where every element is the same, in this case `0`.


```r
Z = matrix(0, 2, 4)
Z
```

```
##      [,1] [,2] [,3] [,4]
## [1,]    0    0    0    0
## [2,]    0    0    0    0
```

Like vectors, matrices can be subsetted using square brackets, `[]`. However, since matrices are two-dimensional, we need to specify both a row and a column when subsetting.


```r
X
```

```
##      [,1] [,2] [,3]
## [1,]    1    4    7
## [2,]    2    5    8
## [3,]    3    6    9
```

```r
X[1, 2]
```

```
## [1] 4
```

Here we accessed the element in the first row and the second column. We could also subset an entire row or column.


```r
X[1, ]
```

```
## [1] 1 4 7
```

```r
X[, 2]
```

```
## [1] 4 5 6
```

We can also use vectors to subset more than one row or column at a time. Here we subset to the first and third column of the second row.


```r
X[2, c(1, 3)]
```

```
## [1] 2 8
```

Matrices can also be created by combining vectors as columns, using `cbind`, or combining vectors as rows, using `rbind`.


```r
x = 1:9
rev(x)
```

```
## [1] 9 8 7 6 5 4 3 2 1
```

```r
rep(1, 9)
```

```
## [1] 1 1 1 1 1 1 1 1 1
```


```r
cbind(x, rev(x), rep(1, 9))
```

```
##       x    
##  [1,] 1 9 1
##  [2,] 2 8 1
##  [3,] 3 7 1
##  [4,] 4 6 1
##  [5,] 5 5 1
##  [6,] 6 4 1
##  [7,] 7 3 1
##  [8,] 8 2 1
##  [9,] 9 1 1
```


```r
rbind(x, rev(x), rep(1, 9))
```

```
##   [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
## x    1    2    3    4    5    6    7    8    9
##      9    8    7    6    5    4    3    2    1
##      1    1    1    1    1    1    1    1    1
```

`R` can then be used to perform matrix calculations.


```r
x = 1:9
y = 9:1
X = matrix(x, 3, 3)
Y = matrix(y, 3, 3)
X
```

```
##      [,1] [,2] [,3]
## [1,]    1    4    7
## [2,]    2    5    8
## [3,]    3    6    9
```

```r
Y
```

```
##      [,1] [,2] [,3]
## [1,]    9    6    3
## [2,]    8    5    2
## [3,]    7    4    1
```


```r
X + Y
```

```
##      [,1] [,2] [,3]
## [1,]   10   10   10
## [2,]   10   10   10
## [3,]   10   10   10
```

```r
X - Y
```

```
##      [,1] [,2] [,3]
## [1,]   -8   -2    4
## [2,]   -6    0    6
## [3,]   -4    2    8
```

```r
X * Y
```

```
##      [,1] [,2] [,3]
## [1,]    9   24   21
## [2,]   16   25   16
## [3,]   21   24    9
```

```r
X / Y
```

```
##           [,1]      [,2]     [,3]
## [1,] 0.1111111 0.6666667 2.333333
## [2,] 0.2500000 1.0000000 4.000000
## [3,] 0.4285714 1.5000000 9.000000
```

Note that `X * Y` is not matrix multiplication. It is element by element multiplication. (Same for `X / Y`). Instead, matrix multiplication uses `%*%`. Other matrix functions include `t()` which gives the transpose of a matrix and `solve()` which returns the inverse of a square matrix if it is invertible.


```r
X %*% Y
```

```
##      [,1] [,2] [,3]
## [1,]   90   54   18
## [2,]  114   69   24
## [3,]  138   84   30
```

```r
t(X)
```

```
##      [,1] [,2] [,3]
## [1,]    1    2    3
## [2,]    4    5    6
## [3,]    7    8    9
```


```r
Z = matrix(c(9, 2, -3, 2, 4, -2, -3, -2, 16), 3, byrow = TRUE)
Z
```

```
##      [,1] [,2] [,3]
## [1,]    9    2   -3
## [2,]    2    4   -2
## [3,]   -3   -2   16
```

```r
solve(Z)
```

```
##             [,1]        [,2]       [,3]
## [1,]  0.12931034 -0.05603448 0.01724138
## [2,] -0.05603448  0.29094828 0.02586207
## [3,]  0.01724138  0.02586207 0.06896552
```

`R` has a number of matrix specific functions for obtaining dimension and summary information.


```r
X = matrix(1:6, 2, 3)
X
```

```
##      [,1] [,2] [,3]
## [1,]    1    3    5
## [2,]    2    4    6
```

```r
dim(X)
```

```
## [1] 2 3
```

```r
rowSums(X)
```

```
## [1]  9 12
```

```r
colSums(X)
```

```
## [1]  3  7 11
```

```r
rowMeans(X)
```

```
## [1] 3 4
```

```r
colMeans(X)
```

```
## [1] 1.5 3.5 5.5
```

The `diag()` function can be used in a number of ways. We can extract the diagonal of a matrix.


```r
diag(Z)
```

```
## [1]  9  4 16
```

Or create a matrix with specified elements on the diagonal. (And `0` on the off-diagonals.)


```r
diag(1:5)
```

```
##      [,1] [,2] [,3] [,4] [,5]
## [1,]    1    0    0    0    0
## [2,]    0    2    0    0    0
## [3,]    0    0    3    0    0
## [4,]    0    0    0    4    0
## [5,]    0    0    0    0    5
```

Or, lastly, create a square matrix of a certain dimension with `1` for every element of the diagonal and `0` for the off-diagonals.


```r
diag(5)
```

```
##      [,1] [,2] [,3] [,4] [,5]
## [1,]    1    0    0    0    0
## [2,]    0    1    0    0    0
## [3,]    0    0    1    0    0
## [4,]    0    0    0    1    0
## [5,]    0    0    0    0    1
```

#### Calculations with Vectors and Matrices {-}

Certain operations in `R`, for example `%*%` have different behavior on vectors and matrices. To illustrate this, we will first create two vectors.


```r
a_vec = c(1, 2, 3)
b_vec = c(2, 2, 2)
```

Note that these are indeed vectors. They are not matrices.


```r
c(is.vector(a_vec), is.vector(b_vec))
```

```
## [1] TRUE TRUE
```

```r
c(is.matrix(a_vec), is.matrix(b_vec))
```

```
## [1] FALSE FALSE
```

When this is the case, the `%*%` operator is used to calculate the **dot product**, also know as the **inner product** of the two vectors.

The dot product of vectors $\boldsymbol{a} = \lbrack a_1, a_2, \cdots a_n \rbrack$ and $\boldsymbol{b} = \lbrack b_1, b_2, \cdots b_n \rbrack$ is defined to be

\[
\boldsymbol{a} \cdot \boldsymbol{b} = \sum_{i = 1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots a_n b_n.
\]


```r
a_vec %*% b_vec # inner product
```

```
##      [,1]
## [1,]   12
```

```r
a_vec %o% b_vec # outer product
```

```
##      [,1] [,2] [,3]
## [1,]    2    2    2
## [2,]    4    4    4
## [3,]    6    6    6
```

The `%o%` operator is used to calculate the **outer product** of the two vectors.

When vectors are coerced to become matrices, they are column vectors. So a vector of length $n$ becomes an $n \times 1$ matrix after coercion.


```r
as.matrix(a_vec)
```

```
##      [,1]
## [1,]    1
## [2,]    2
## [3,]    3
```

If we use the `%*%` operator on matrices, `%*%` again performs the expected matrix multiplication. So you might expect the following to produce an error, because the dimensions are incorrect.


```r
as.matrix(a_vec) %*% b_vec
```

```
##      [,1] [,2] [,3]
## [1,]    2    2    2
## [2,]    4    4    4
## [3,]    6    6    6
```

At face value this is a $3 \times 1$ matrix, multiplied by a $3 \times 1$ matrix. However, when `b_vec` is automatically coerced to be a matrix, `R` decided to make it a "row vector", a $1 \times 3$ matrix, so that the multiplication has conformable dimensions.

If we had coerced both, then `R` would produce an error.


```r
as.matrix(a_vec) %*% as.matrix(b_vec)
```

Another way to calculate a *dot product* is with the `crossprod()` function. Given two vectors, the `crossprod()` function calculates their dot product. The function has a rather misleading name.


```r
crossprod(a_vec, b_vec)  # inner product
```

```
##      [,1]
## [1,]   12
```

```r
tcrossprod(a_vec, b_vec)  # outer product
```

```
##      [,1] [,2] [,3]
## [1,]    2    2    2
## [2,]    4    4    4
## [3,]    6    6    6
```

These functions could be very useful later. When used with matrices $X$ and $Y$ as arguments, it calculates

\[
X^\top Y.
\]

When dealing with linear models, the calculation

\[
X^\top X
\]

is used repeatedly.


```r
C_mat = matrix(c(1, 2, 3, 4, 5, 6), 2, 3)
D_mat = matrix(c(2, 2, 2, 2, 2, 2), 2, 3)
```

This is useful both as a shortcut for a frequent calculation and as a more efficient implementation than using `t()` and `%*%`.


```r
crossprod(C_mat, D_mat)
```

```
##      [,1] [,2] [,3]
## [1,]    6    6    6
## [2,]   14   14   14
## [3,]   22   22   22
```

```r
t(C_mat) %*% D_mat
```

```
##      [,1] [,2] [,3]
## [1,]    6    6    6
## [2,]   14   14   14
## [3,]   22   22   22
```

```r
all.equal(crossprod(C_mat, D_mat), t(C_mat) %*% D_mat)
```

```
## [1] TRUE
```


```r
crossprod(C_mat, C_mat)
```

```
##      [,1] [,2] [,3]
## [1,]    5   11   17
## [2,]   11   25   39
## [3,]   17   39   61
```

```r
t(C_mat) %*% C_mat
```

```
##      [,1] [,2] [,3]
## [1,]    5   11   17
## [2,]   11   25   39
## [3,]   17   39   61
```

```r
all.equal(crossprod(C_mat, C_mat), t(C_mat) %*% C_mat)
```

```
## [1] TRUE
```

### Data Frames

We have previously seen vectors and matrices for storing data as we introduced `R`. We will now introduce a **data frame** which will be the most common way that we store and interact with data in this course.


```r
example_data = data.frame(x = c(1, 3, 5, 7, 9, 1, 3, 5, 7, 9),
                          y = rep("Hello", 10),
                          z = rep(c("TRUE", "FALSE"), 5))
```

Unlike a matrix, which can be thought of as a vector rearranged into rows and columns, a data frame is not required to have the same data type for each element. A data frame is a **list** of vectors. So, each vector must contain the same data type, but the different vectors can store different data types. 


```r
example_data
```

```
##    x     y     z
## 1  1 Hello  TRUE
## 2  3 Hello FALSE
## 3  5 Hello  TRUE
## 4  7 Hello FALSE
## 5  9 Hello  TRUE
## 6  1 Hello FALSE
## 7  3 Hello  TRUE
## 8  5 Hello FALSE
## 9  7 Hello  TRUE
## 10 9 Hello FALSE
```

The `data.frame()` function above is one way to create a data frame. We can also import data from various file types in into `R`, as well as use data stored in packages.



[The example data above can also be found here as a .csv file.](data/example_data.csv) To read this data into `R`, we would use the `read_csv()` function from the `readr` package. Note that `R` has a built in function `read.csv()` that operates very similarly. The `readr` function `read_csv()` has a number of advantages. For example, it is much faster reading larger data. [It also uses the `tibble` package to read the data as a tibble.](https://cran.r-project.org/web/packages/tibble/vignettes/tibble.html)


```r
library(readr)
example_data_from_csv = read_csv("data/example_data.csv")
```

This particular line of code assumes that the file `example_data.csv` exists in a folder called `data` in your current working directory.


```r
example_data_from_csv
```

```
## # A tibble: 10 × 3
##        x     y     z
##    <int> <chr> <lgl>
## 1      1 Hello  TRUE
## 2      3 Hello FALSE
## 3      5 Hello  TRUE
## 4      7 Hello FALSE
## 5      9 Hello  TRUE
## 6      1 Hello FALSE
## 7      3 Hello  TRUE
## 8      5 Hello FALSE
## 9      7 Hello  TRUE
## 10     9 Hello FALSE
```

A tibble is simply a data frame that prints with sanity. Notice in the output above that we are given additional information such as dimension and variable type.

The `as_tibble()` function can be used to coerce a regular data frame to a tibble.


```r
library(tibble)
example_data = as_tibble(example_data)
example_data
```

```
## # A tibble: 10 × 3
##        x      y      z
##    <dbl> <fctr> <fctr>
## 1      1  Hello   TRUE
## 2      3  Hello  FALSE
## 3      5  Hello   TRUE
## 4      7  Hello  FALSE
## 5      9  Hello   TRUE
## 6      1  Hello  FALSE
## 7      3  Hello   TRUE
## 8      5  Hello  FALSE
## 9      7  Hello   TRUE
## 10     9  Hello  FALSE
```

Alternatively, we could use the "Import Dataset" feature in RStudio which can be found in the environment window. (By default, the top-right pane of RStudio.)  Once completed, this process will automatically generate the code to import a file. The resulting code will be shown in the console window. In recent versions of RStudio, `read_csv()` is used by default, thus reading in a tibble.

Earlier we looked at installing packages, in particular the `ggplot2` package. (A package for visualization. While not necessary for this course, it is quickly growing in popularity.) 


```r
library(ggplot2)
```

Inside the `ggplot2` package is a dataset called `mpg`. By loading the package using the `library()` function, we can now access `mpg`.

When using data from inside a package, there are three things we would generally like to do:

- Look at the raw data.
- Understand the data. (Where did it come from? What are the variables? Etc.)
- Visualize the data.

To look at the data, we have two useful commands: `head()` and `str()`.


```r
head(mpg, n = 10)
```

```
## # A tibble: 10 × 11
##    manufacturer      model displ  year   cyl      trans   drv   cty   hwy
##           <chr>      <chr> <dbl> <int> <int>      <chr> <chr> <int> <int>
## 1          audi         a4   1.8  1999     4   auto(l5)     f    18    29
## 2          audi         a4   1.8  1999     4 manual(m5)     f    21    29
## 3          audi         a4   2.0  2008     4 manual(m6)     f    20    31
## 4          audi         a4   2.0  2008     4   auto(av)     f    21    30
## 5          audi         a4   2.8  1999     6   auto(l5)     f    16    26
## 6          audi         a4   2.8  1999     6 manual(m5)     f    18    26
## 7          audi         a4   3.1  2008     6   auto(av)     f    18    27
## 8          audi a4 quattro   1.8  1999     4 manual(m5)     4    18    26
## 9          audi a4 quattro   1.8  1999     4   auto(l5)     4    16    25
## 10         audi a4 quattro   2.0  2008     4 manual(m6)     4    20    28
## # ... with 2 more variables: fl <chr>, class <chr>
```

The function `head()` will display the first `n` observations of the data frame. The `head()` function was more useful before tibbles. Notice that `mpg` is a tibble already, so the output from `head()` indicates there are only `10` observations. Note that this applies to `head(mpg, n = 10)` and not `mpg` itself. Also note that tibbles print a limited number of rows and columns by default. The last line of the printed output indicates with rows and columns were omitted.


```r
mpg
```

```
## # A tibble: 234 × 11
##    manufacturer      model displ  year   cyl      trans   drv   cty   hwy
##           <chr>      <chr> <dbl> <int> <int>      <chr> <chr> <int> <int>
## 1          audi         a4   1.8  1999     4   auto(l5)     f    18    29
## 2          audi         a4   1.8  1999     4 manual(m5)     f    21    29
## 3          audi         a4   2.0  2008     4 manual(m6)     f    20    31
## 4          audi         a4   2.0  2008     4   auto(av)     f    21    30
## 5          audi         a4   2.8  1999     6   auto(l5)     f    16    26
## 6          audi         a4   2.8  1999     6 manual(m5)     f    18    26
## 7          audi         a4   3.1  2008     6   auto(av)     f    18    27
## 8          audi a4 quattro   1.8  1999     4 manual(m5)     4    18    26
## 9          audi a4 quattro   1.8  1999     4   auto(l5)     4    16    25
## 10         audi a4 quattro   2.0  2008     4 manual(m6)     4    20    28
## # ... with 224 more rows, and 2 more variables: fl <chr>, class <chr>
```

The function `str()` will display the "structure" of the data frame. It will display the number of **observations** and **variables**, list the variables, give the type of each variable, and show some elements of each variable. This information can also be found in the "Environment" window in RStudio.


```r
str(mpg)
```

```
## Classes 'tbl_df', 'tbl' and 'data.frame':	234 obs. of  11 variables:
##  $ manufacturer: chr  "audi" "audi" "audi" "audi" ...
##  $ model       : chr  "a4" "a4" "a4" "a4" ...
##  $ displ       : num  1.8 1.8 2 2 2.8 2.8 3.1 1.8 1.8 2 ...
##  $ year        : int  1999 1999 2008 2008 1999 1999 2008 1999 1999 2008 ...
##  $ cyl         : int  4 4 4 4 6 6 6 4 4 4 ...
##  $ trans       : chr  "auto(l5)" "manual(m5)" "manual(m6)" "auto(av)" ...
##  $ drv         : chr  "f" "f" "f" "f" ...
##  $ cty         : int  18 21 20 21 16 18 18 18 16 20 ...
##  $ hwy         : int  29 29 31 30 26 26 27 26 25 28 ...
##  $ fl          : chr  "p" "p" "p" "p" ...
##  $ class       : chr  "compact" "compact" "compact" "compact" ...
```

It is important to note that while matrices have rows and columns, data frames (tibbles) instead have observations and variables. When displayed in the console or viewer, each row is an observation and each column is a variable. However generally speaking, their order does not matter, it is simply a side-effect of how the data was entered or stored.

In this dataset an observation is for a particular model-year of a car, and the variables describe attributes of the car, for example its highway fuel efficiency.

To understand more about the data set, we use the `?` operator to pull up the documentation for the data.


```r
?mpg
```

`R` has a number of functions for quickly working with and extracting basic information from data frames. To quickly obtain a vector of the variable names, we use the `names()` function.


```r
names(mpg)
```

```
##  [1] "manufacturer" "model"        "displ"        "year"        
##  [5] "cyl"          "trans"        "drv"          "cty"         
##  [9] "hwy"          "fl"           "class"
```

To access one of the variables as a vector, we use the `$` operator.


```r
mpg$year
```

```
##   [1] 1999 1999 2008 2008 1999 1999 2008 1999 1999 2008 2008 1999 1999 2008
##  [15] 2008 1999 2008 2008 2008 2008 2008 1999 2008 1999 1999 2008 2008 2008
##  [29] 2008 2008 1999 1999 1999 2008 1999 2008 2008 1999 1999 1999 1999 2008
##  [43] 2008 2008 1999 1999 2008 2008 2008 2008 1999 1999 2008 2008 2008 1999
##  [57] 1999 1999 2008 2008 2008 1999 2008 1999 2008 2008 2008 2008 2008 2008
##  [71] 1999 1999 2008 1999 1999 1999 2008 1999 1999 1999 2008 2008 1999 1999
##  [85] 1999 1999 1999 2008 1999 2008 1999 1999 2008 2008 1999 1999 2008 2008
##  [99] 2008 1999 1999 1999 1999 1999 2008 2008 2008 2008 1999 1999 2008 2008
## [113] 1999 1999 2008 1999 1999 2008 2008 2008 2008 2008 2008 2008 1999 1999
## [127] 2008 2008 2008 2008 1999 2008 2008 1999 1999 1999 2008 1999 2008 2008
## [141] 1999 1999 1999 2008 2008 2008 2008 1999 1999 2008 1999 1999 2008 2008
## [155] 1999 1999 1999 2008 2008 1999 1999 2008 2008 2008 2008 1999 1999 1999
## [169] 1999 2008 2008 2008 2008 1999 1999 1999 1999 2008 2008 1999 1999 2008
## [183] 2008 1999 1999 2008 1999 1999 2008 2008 1999 1999 2008 1999 1999 1999
## [197] 2008 2008 1999 2008 1999 1999 2008 1999 1999 2008 2008 1999 1999 2008
## [211] 2008 1999 1999 1999 1999 2008 2008 2008 2008 1999 1999 1999 1999 1999
## [225] 1999 2008 2008 1999 1999 2008 2008 1999 1999 2008
```

```r
mpg$hwy
```

```
##   [1] 29 29 31 30 26 26 27 26 25 28 27 25 25 25 25 24 25 23 20 15 20 17 17
##  [24] 26 23 26 25 24 19 14 15 17 27 30 26 29 26 24 24 22 22 24 24 17 22 21
##  [47] 23 23 19 18 17 17 19 19 12 17 15 17 17 12 17 16 18 15 16 12 17 17 16
##  [70] 12 15 16 17 15 17 17 18 17 19 17 19 19 17 17 17 16 16 17 15 17 26 25
##  [93] 26 24 21 22 23 22 20 33 32 32 29 32 34 36 36 29 26 27 30 31 26 26 28
## [116] 26 29 28 27 24 24 24 22 19 20 17 12 19 18 14 15 18 18 15 17 16 18 17
## [139] 19 19 17 29 27 31 32 27 26 26 25 25 17 17 20 18 26 26 27 28 25 25 24
## [162] 27 25 26 23 26 26 26 26 25 27 25 27 20 20 19 17 20 17 29 27 31 31 26
## [185] 26 28 27 29 31 31 26 26 27 30 33 35 37 35 15 18 20 20 22 17 19 18 20
## [208] 29 26 29 29 24 44 29 26 29 29 29 29 23 24 44 41 29 26 28 29 29 29 28
## [231] 29 26 26 26
```

We can use the `dim()`, `nrow()` and `ncol()` functions to obtain information about the dimension of the data frame.


```r
dim(mpg)
```

```
## [1] 234  11
```

```r
nrow(mpg)
```

```
## [1] 234
```

```r
ncol(mpg)
```

```
## [1] 11
```

Here `nrow()` is also the number of observations, which in most cases is the *sample size*.

Subsetting data frames can work much like subsetting matrices using square brackets, `[,]`. Here, we find fuel efficient vehicles earning over 35 miles per gallon and only display `manufacturer`, `model` and `year`.


```r
mpg[mpg$hwy > 35, c("manufacturer", "model", "year")]
```

```
## # A tibble: 6 × 3
##   manufacturer      model  year
##          <chr>      <chr> <int>
## 1        honda      civic  2008
## 2        honda      civic  2008
## 3       toyota    corolla  2008
## 4   volkswagen      jetta  1999
## 5   volkswagen new beetle  1999
## 6   volkswagen new beetle  1999
```

An alternative would be to use the `subset()` function, which has a much more readable syntax.


```r
subset(mpg, subset = hwy > 35, select = c("manufacturer", "model", "year"))
```

Lastly, we could use the `filter` and `select` functions from the `dplyr` package which introduces the `%>%` operator from the `magrittr` package. This is not necessary for this course, however the `dplyr` package is something you should be aware of as it is becoming a popular tool in the `R` world.


```r
library(dplyr)
mpg %>% filter(hwy > 35) %>% select(manufacturer, model, year)
```

All three approaches produce the same results. Which you use will be largely based on a given situation as well as user preference.

### Plotting

Now that we have some data to work with, and we have learned about the data at the most basic level, our next tasks is to visualize the data. Often, a proper visualization can illuminate features of the data that can inform further analysis.

We will look at three methods of visualizing data that we will use throughout the course:

- Histograms
- Boxplots
- Scatterplots

#### Histograms

When visualizing a single numerical variable, a **histogram** will be our go-to tool, which can be created in `R` using the `hist()` function.


```r
hist(mpg$cty)
```

![](02-r-intro_files/figure-latex/unnamed-chunk-63-1.pdf)<!-- --> 

The histogram function has a number of parameters which can be changed to make our plot look much nicer. Use the `?` operator to read the documentation for the `hist()` to see a full list of these parameters.


```r
hist(mpg$cty,
     xlab   = "Miles Per Gallon (City)",
     main   = "Histogram of MPG (City)",
     breaks = 12,
     col    = "dodgerblue",
     border = "darkorange")
```

![](02-r-intro_files/figure-latex/unnamed-chunk-64-1.pdf)<!-- --> 

Importantly, you should always be sure to label your axes and give the plot a title. The argument `breaks` is specific to `hist()`. Entering an integer will give a suggestion to `R` for how many bars to use for the histogram. By default `R` will attempt to intelligently guess a good number of `breaks`, but as we can see here, it is sometimes useful to modify this yourself.

#### Boxplots

To visualize the relationship between a numerical and categorical variable, we will use a **boxplot**. In the `mpg` dataset, the `drv` variable takes a small, finite number of values. A car can only be front wheel drive, 4 wheel drive, or rear wheel drive.


```r
unique(mpg$drv)
```

```
## [1] "f" "4" "r"
```

First note that we can use a single boxplot as an alternative to a histogram for visualizing a single numerical variable. To do so in `R`, we use the `boxplot()` function.


```r
boxplot(mpg$hwy)
```

![](02-r-intro_files/figure-latex/unnamed-chunk-66-1.pdf)<!-- --> 

However, more often we will use boxplots to compare a numerical variable for different values of a categorical variable.


```r
boxplot(hwy ~ drv, data = mpg)
```

![](02-r-intro_files/figure-latex/unnamed-chunk-67-1.pdf)<!-- --> 

Here used the `boxplot()` command to create side-by-side boxplots. However, since we are now dealing with two variables, the syntax has changed. The `R` syntax `hwy ~ drv, data = mpg` reads "Plot the `hwy` variable against the `drv` variable using the dataset `mpg`." We see the use of a `~` (which specifies a formula) and also a `data = ` argument. This will be a syntax that is common to many functions we will use in this course. 


```r
boxplot(hwy ~ drv, data = mpg,
     xlab   = "Drivetrain (f = FWD, r = RWD, 4 = 4WD)",
     ylab   = "Miles Per Gallon (Highway)",
     main   = "MPG (Highway) vs Drivetrain",
     pch    = 20,
     cex    = 2,
     col    = "darkorange",
     border = "dodgerblue")
```

![](02-r-intro_files/figure-latex/unnamed-chunk-68-1.pdf)<!-- --> 

Again, `boxplot()` has a number of additional arguments which have the ability to make our plot more visually appealing.

#### Scatterplots

Lastly, to visualize the relationship between two numeric variables we will use a **scatterplot**. This can be done with the `plot()` function and the `~` syntax we just used with a boxplot. (The function `plot()` can also be used more generally; see the documentation for details.)


```r
plot(hwy ~ displ, data = mpg)
```

![](02-r-intro_files/figure-latex/unnamed-chunk-69-1.pdf)<!-- --> 


```r
plot(hwy ~ displ, data = mpg,
     xlab = "Engine Displacement (in Liters)",
     ylab = "Miles Per Gallon (Highway)",
     main = "MPG (Highway) vs Engine Displacement",
     pch  = 20,
     cex  = 2,
     col  = "dodgerblue")
```

![](02-r-intro_files/figure-latex/unnamed-chunk-70-1.pdf)<!-- --> 

### Distributions

When working with different statistical distributions, we often want to make probabilistic statements based on the distribution.

We typically want to know one of four things:

* The density (pdf) at a particular value.
* The distribution (cdf) at a particular value.
* The quantile value corresponding to a particular probability.
* A random draw of values from a particular distribution.

This used to be done with statistical tables printed in the back of textbooks. Now, `R` has functions for obtaining density, distribution, quantile and random values.

The general naming structure of the relevant `R` functions is:

* `dname` calculates density (pdf) at input `x`.
* `pname` calculates distribution (cdf) at input `x`.
* `qname` calculates the quantile at an input probability.
* `rname` generates a random draw from a particular distribution.

Note that `name` represents the name of the given distribution.

For example, consider a random variable $X$ which is $N(\mu = 2, \sigma^2 = 25)$. (Note, we are parameterizing using the variance $\sigma^2$. `R` however uses the standard deviation.)

To calculate the value of the pdf at `x = 3`, that is, the height of the curve at `x = 3`, use:


```r
dnorm(x = 3, mean = 2, sd = 5)
```

```
## [1] 0.07820854
```

To calculate the value of the cdf at `x = 3`, that is, $P(X \leq 3)$, the probability that $X$ is less than or equal to `3`, use:


```r
pnorm(q = 3, mean = 2, sd = 5)
```

```
## [1] 0.5792597
```

Or, to calculate the quantile for probability 0.975, use:


```r
qnorm(p = 0.975, mean = 2, sd = 5)
```

```
## [1] 11.79982
```

Lastly, to generate a random sample of size `n = 10`, use:


```r
rnorm(n = 10, mean = 2, sd = 5)
```

```
##  [1]  0.2089128  1.8000486 -0.6957943  4.9121413  1.6498175 11.1722064
##  [7]  6.2486654 -1.6989696  1.5267841  6.9303569
```

These functions exist for many other distributions, including but not limited to:

| Command  | Distribution |
|----------|--------------|
| `*binom` | Binomial     |
| `*t`     | t            |
| `*pois`  | Poisson      |
| `*f`     | F            |
| `*chisq` | Chi-Squared  |

Where `*` can be `d`, `p`, `q`, and `r`. Each distribution will have its own set of parameters which need to be passed to the functions as arguments. For example, `dbinom()` would not have arguments for `mean` and `sd`, since those are not parameters of the distribution. Instead a binomial distribution is usually parameterized by $n$ and $p$, however `R` chooses to call them something else. To find the names that `R` uses we would use `?dbinom` and see that `R` instead calls the arguments `size` and `prob`. For example:


```r
dbinom(x = 6, size = 10, prob = 0.75)
```

```
## [1] 0.145998
```

Also note that, when using the `dname` functions with discrete distributions, they are the pmf of the distribution. For example, the above command is $P(Y = 6)$ if $Y \sim b(n = 10, p = 0.75)$. (The probability of flipping an unfair coin `10` times and seeing `6` heads, if the probability of heads is `0.75`.)

## Programming Basics

### Logical Operators

| Operator | Summary               | Example               | Result |
|----------|-----------------------|-----------------------|--------|
| `x < y`  | `x` less than `y`                | `3 < 42`               | TRUE               |
| `x > y`  | `x` greater than `y`             | `3 > 42`               | FALSE               |
| `x <= y` | `x` less than or equal to `y`    | `3 <= 42`              | TRUE              |
| `x >= y` | `x` greater than or equal to `y` | `3 >= 42`              | FALSE              |
| `x == y` | `x`equal to `y`                  | `3 == 42`              | FALSE              |
| `x != y` | `x` not equal to `y`             | `3 != 42`              | TRUE              |
| `!x`     | not `x`                          | `!(3 > 42)`            | TRUE            |
| `x | y`  | `x` or `y`                       | `(3 > 42) | TRUE`      | TRUE      |
| `x & y`  | `x` and `y`                      | `(3 < 4) & ( 42 > 13)` | TRUE |

In `R`, logical operators are vectorized. To demonstrate this, we will use the following height and weight data.


```r
heights = c(110, 120, 115, 136, 205, 156, 175)
weights = c(64, 67, 62, 60, 77, 70, 66)
```

First, using the `<` operator, when can find which `heights` are less than `121`. Further, we could also find which `heights` are less than `121` or exactly equal to `156.`


```r
heights < 121
```

```
## [1]  TRUE  TRUE  TRUE FALSE FALSE FALSE FALSE
```

```r
heights < 121 | heights == 156
```

```
## [1]  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE
```

Often, a vector of logical values is useful for subsetting a vector. For example, we can find the `heights` that are larger than `150`. We can then use the resulting vector to subset the `heights` vector, thus actually returning the `heights` that are above `150`, instead of a vector of which values are above `150`. Here we also obtain the `weights` corresponding to `heights` above `150`.


```r
heights > 150
```

```
## [1] FALSE FALSE FALSE FALSE  TRUE  TRUE  TRUE
```

```r
heights[heights > 150]
```

```
## [1] 205 156 175
```

```r
weights[heights > 150]
```

```
## [1] 77 70 66
```

When comparing vectors, be sure you are comparing vectors of the same length.


```r
a = 1:10
b = 2:4
a < b
```

```
## Warning in a < b: longer object length is not a multiple of shorter object
## length
```

```
##  [1]  TRUE  TRUE  TRUE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
```

What happened here? `R` still performed the operation, but it also gives us a warning. (To perform the operation, `R` automatically made `b` longer by repeating `b` as needed.)

The one exception to this behavior is comparing to a vector of length `1`. `R` does not warn us in this case, as comparing each value of a vector to a single value is a common operation that is usually reasonable to perform.


```r
a > 5
```

```
##  [1] FALSE FALSE FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
```

Often we will want to convert `TRUE` and `FALSE` values to `1` and `0`. When performing mathematical operations on `TRUE` and `FALSE`, this is done automatically through type coercion.


```r
5 + (a > 5)
```

```
##  [1] 5 5 5 5 5 6 6 6 6 6
```

By calling `sum()` on a vector of logical values, we can essentially count the number of `TRUE` values.


```r
sum(a > 5)
```

```
## [1] 5
```

Here we count the elements of `a` that are larger than `5.` This is an extremely useful feature.

### Control Flow

In `R`, the if/else syntax is:


```r
if (...) {
  some R code
} else {
  more R code
}
```

For example,


```r
x = 1
y = 3
if (x > y) {
  z = x * y
  print("x is larger than y")
} else {
  z = x + 5 * y
  print("x is less than or equal to y")
}
```

```
## [1] "x is less than or equal to y"
```

```r
z
```

```
## [1] 16
```

`R` also has a special function `ifelse()` which is very useful. It returns one of two specified values based on a conditional statement.


```r
ifelse(4 > 3, 1, 0)
```

```
## [1] 1
```

The real power of `ifelse()` comes from its ability to be applied to vectors.


```r
fib = c(1, 1, 2, 3, 5, 8, 13, 21)
ifelse(fib > 6, "Foo", "Bar")
```

```
## [1] "Bar" "Bar" "Bar" "Bar" "Bar" "Foo" "Foo" "Foo"
```

Now a `for` loop example,


```r
x = 11:15
for (i in 1:5) {
  x[i] = x[i] * 2
}

x
```

```
## [1] 22 24 26 28 30
```

Note that this `for` loop is very normal in many programming languages, but not in `R`. In `R` we would not use a loop, instead we would simply use a vectorized operation.


```r
x = 11:15
x = x * 2
x
```

```
## [1] 22 24 26 28 30
```

### Functions

So far we have been using functions, but haven't actually discussed some of their details.


```r
function_name(arg1 = 10, arg2 = 20)
```

To use a function, you simply type its name, followed by an open parenthesis, then specify values of its arguments, then finish with a closing parenthesis. 

An **argument** is a variable which is used in the body of the function. Specifying the values of the arguments is essentially providing the inputs to the function.

We can also write our own functions in `R`. For example, we often like to "standardize" variables, that is, subtracting the sample mean, and dividing by the sample standard deviation.

\[
\frac{x - \bar{x}}{s}
\]

In `R` we would write a function to do this. When writing a function, there are three thing you must do.

- Give the function a name. Preferably something that is short, but descriptive.
- Specify the arguments using `function()`
- Write the body of the function within curly braces, `{}`.


```r
standardize = function(x) {
  m = mean(x)
  std = sd(x)
  result = (x - m) / std
  result
}
```

Here the name of the function is `standardize`, and the function has a single argument `x` which is used in the body of function. Note that the output of the final line of the body is what is returned by the function. In this case the function returns the vector stored in the variable `results`.

To test our function, we will take a random sample of size `n = 10` from a normal distribution with a mean of `2` and a standard deviation of `5`.


```r
(test_sample = rnorm(n = 10, mean = 2, sd = 5))
```

```
##  [1]  7.8832249  4.5095260  1.5534411  6.1356400 -0.8878737  0.8397304
##  [7]  4.2525988  6.9366983  6.3495624  1.6175390
```

```r
standardize(x = test_sample)
```

```
##  [1]  1.3331154  0.1985835 -0.7955103  0.7454249 -1.6164934 -1.0355221
##  [7]  0.1121821  1.0148106  0.8173643 -0.7739550
```

This function could be written much more succinctly, simply performing all the operations on one line and immediately returning the result, without storing any of the intermediate results.


```r
standardize = function(x) {
  (x - mean(x)) / sd(x)
}
```

When specifying arguments, you can provide default arguments.


```r
power_of_num = function(num, power = 2) {
  num ^ power
}
```

Let's look at a number of ways that we could run this function to perform the operation `10^2` resulting in `100`.


```r
power_of_num(10)
```

```
## [1] 100
```

```r
power_of_num(10, 2)
```

```
## [1] 100
```

```r
power_of_num(num = 10, power = 2)
```

```
## [1] 100
```

```r
power_of_num(power = 2, num = 10)
```

```
## [1] 100
```

Note that without using the argument names, the order matters. The following code will not evaluate to the same output as the previous example.


```r
power_of_num(2, 10)
```

```
## [1] 1024
```

Also, the following line of code would produce an error since arguments without a default value must be specified.


```r
power_of_num(power = 5)
```

To further illustrate a function with a default argument, we will write a function that calculates sample variance two ways.

By default, is will calculate the unbiased estimate of $\sigma^2$, which we will call $s^2$.

\[
s^2 = \frac{1}{n - 1}\sum_{i=1}^{n}(x - \bar{x})^2
\]

It will also have the ability to return the biased estimate (based on maximum likelihood) which we will call $\hat{\sigma}^2$.

\[
\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x - \bar{x})^2
\]


```r
get_var = function(x, biased = FALSE) {
  n = length(x) - 1 * !biased
  (1 / n) * sum((x - mean(x)) ^ 2)
}
```


```r
get_var(test_sample)
```

```
## [1] 8.842583
```

```r
get_var(test_sample, biased = FALSE)
```

```
## [1] 8.842583
```

```r
var(test_sample)
```

```
## [1] 8.842583
```

We see the function is working as expected, and when returning the unbiased estimate it matches `R`'s built in function `var()`. Finally, let's examine the biased estimate of $\sigma^2$.


```r
get_var(test_sample, biased = TRUE)
```

```
## [1] 7.958325
```

## Hypothesis Tests in `R`

A prerequisite for STAT 420 is an understanding of the basics of hypothesis testing. Recall the basic structure of hypothesis tests:

- An overall model and related assumptions are made. (The most common being observations following a normal distribution.)
- The **null** ($H_{0}$) and **alternative** ($H_{1}$ or $H_{A}$) hypothesis are specified. Usually the null specifies a particular value of a parameter.
- With given data, the **value** of the *test statistic* is calculated.
- Under the general assumptions, as well as assuming the null hypothesis is true, the **distribution** of the *test statistic* is known.
- Given the distribution and value of the test statistic, as well as the form of the alternative hypothesis, we can calculate a **p-value** of the test.
- Based on the **p-value** and pre-specified level of significance, we make a decision. One of:
    - Fail to reject the null hypothesis.
    - Reject the null hypothesis.
    
We'll do some quick review of two of the most common tests to show how they are performed using `R`.

### One Sample t-Test: Review

Suppose $x_{i} \sim \mathrm{N}(\mu,\sigma^{2})$ and we want to test $H_{0}: \mu = \mu_{0}$ versus $H_{1}: \mu \neq \mu_{0}.$

Assuming $\sigma$ is unknown, we use the one-sample Student's $t$ test statistic:

\[
t = \frac{\bar{x}-\mu_{0}}{s/\sqrt{n}} \sim t_{n-1},
\]

where $\bar{x} = \displaystyle\frac{\sum_{i=1}^{n}x_{i}}{n}$ and $s = \sqrt{\displaystyle\frac{1}{n - 1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$.

A $100(1 - \alpha)$\% confidence interval for $\mu$ is given by,

\[
\bar{x} \pm t_{n-1}(\alpha/2)\frac{s}{\sqrt{n}}
\]

where $t_{n-1}(\alpha/2)$ is the critical value such that $P\left(t>t_{n-1}(\alpha/2)\right) = \alpha/2$ for $n-1$ degrees of freedom.

### One Sample t-Test: Example

Suppose a grocery store sells "16 ounce" boxes of *Captain Crisp* cereal. A random sample of 9 boxes was taken and weighed. The weight in ounces are stored in the data frame `capt_crisp`.


```r
capt_crisp = data.frame(weight = c(15.5, 16.2, 16.1, 15.8, 15.6, 16.0, 15.8, 15.9, 16.2))
```

The company that makes *Captain Crisp* cereal claims that the average weight of a box is at least 16 ounces. We will assume the weight of cereal in a box is normally distributed and use a 0.05 level of significance to test the company's claim.

To test $H_{0}: \mu \geq 16$ versus $H_{1}: \mu < 16$, the test statistic is

\[
t = \frac{\bar{x} - \mu_{0}}{s / \sqrt{n}}
\]

The sample mean $\bar{x}$ and the sample standard deviation $s$ can be easily computed using `R`. We also create variables which store the hypothesized mean and the sample size.


```r
x_bar = mean(capt_crisp$weight)
s     = sd(capt_crisp$weight)
mu_0  = 16
n     = 9
```

We can then easily compute the test statistic.


```r
t = (x_bar - mu_0) / (s / sqrt(n))
t
```

```
## [1] -1.2
```

Under the null hypothesis, the test statistic has a $t$ distribution with $n - 1$ degrees of freedom, in this case 8.

To complete the test, we need to obtain the p-value of the test. Since this is a one-sided test with a less-than alternative, we need to area to the left of -1.2 for a $t$ distribution with 8 degrees of freedom. That is,

\[
P(t_{8} < -1.2)
\]


```r
pt(t, df = n - 1)
```

```
## [1] 0.1322336
```

We now have the p-value of our test, which is greater than our significance level (0.05), so we fail to reject the null hypothesis.

Alternatively, this entire process could have been completed using one line of `R` code.


```r
t.test(x = capt_crisp$weight, mu = 16, alternative = c("less"), conf.level = 0.95)
```

```
## 
## 	One Sample t-test
## 
## data:  capt_crisp$weight
## t = -1.2, df = 8, p-value = 0.1322
## alternative hypothesis: true mean is less than 16
## 95 percent confidence interval:
##      -Inf 16.05496
## sample estimates:
## mean of x 
##      15.9
```

We supply `R` with the data, the hypothesized value of $\mu$, the alternative, and the confidence level. `R` then returns a wealth of information including:

- The value of the test statistic.
- The degrees of freedom of the distribution under the null hypothesis.
- The p-value of the test.
- The confidence interval which corresponds to the test.
- An estimate of $\mu$.

Since the test was one-sided, `R` returned a one-sided confidence interval. If instead we wanted a two-sided interval for the mean weight of boxes of *Captain Crisp* cereal we could modify our code.


```r
capt_test_results = t.test(capt_crisp$weight, mu = 16,
                           alternative = c("two.sided"), conf.level = 0.95)
```

This time we have stored the results. By doing so, we can directly access portions of the output from `t.test()`. To see what information is available we use the `names()` function.


```r
names(capt_test_results)
```

```
## [1] "statistic"   "parameter"   "p.value"     "conf.int"    "estimate"   
## [6] "null.value"  "alternative" "method"      "data.name"
```

We are interested in the confidence interval which is stored in `conf.int`.


```r
capt_test_results$conf.int
```

```
## [1] 15.70783 16.09217
## attr(,"conf.level")
## [1] 0.95
```

Let's check this interval "by hand." The one piece of information we are missing is the critical value, $t_{n-1}(\alpha/2) = t_{8}(0.025)$, which can be calculated in `R` using the `qt()` function.


```r
qt(0.975, df = 8)
```

```
## [1] 2.306004
```

So, the 95\% CI for the mean weight of a cereal box is calculated by plugging into the formula,

\[
\bar{x} \pm t_{n-1}(\alpha/2) \frac{s}{\sqrt{n}}
\]


```r
c(mean(capt_crisp$weight) - qt(0.975, df = 8) * sd(capt_crisp$weight) / sqrt(9),
  mean(capt_crisp$weight) + qt(0.975, df = 8) * sd(capt_crisp$weight) / sqrt(9))
```

```
## [1] 15.70783 16.09217
```

### Two Sample t-Test: Review

Suppose $x_{i} \sim \mathrm{N}(\mu_{x}, \sigma^{2})$ and $y_{i} \sim \mathrm{N}(\mu_{y}, \sigma^{2}).$ 

Want to test $H_{0}: \mu_{x} - \mu_{y} = \mu_{0}$ versus $H_{1}: \mu_{x} - \mu_{y} \neq \mu_{0}.$

Assuming $\sigma$ is unknown, use the two-sample Student's $t$ test statistic:

\[
t = \frac{(\bar{x} - \bar{y})-\mu_{0}}{s_{p}\sqrt{\frac{1}{n}+\frac{1}{m}}} \sim t_{n+m-2},
\]

where $\displaystyle\bar{x}=\frac{\sum_{i=1}^{n}x_{i}}{n}$, $\displaystyle\bar{y}=\frac{\sum_{i=1}^{m}y_{i}}{m}$, and $s_p^2 = \displaystyle\frac{(n-1)s_x^2+(m-1)s_y^2}{n+m-2}$.

A $100(1-\alpha)$\% CI for $\mu_{x}-\mu_{y}$ is given by

\[
(\bar{x} - \bar{y}) \pm t_{n+m-2}(\alpha/2) \left(s_{p}\textstyle\sqrt{\frac{1}{n}+\frac{1}{m}}\right),
\]

where $t_{n+m-2}(\alpha/2)$ is the critical value such that $P\left(t>t_{n+m-2}(\alpha/2)\right)=\alpha/2$.

### Two Sample t-Test: Example

Assume that the distributions of $X$ and $Y$ are $\mathrm{N}(\mu_{1},\sigma^{2})$ and $\mathrm{N}(\mu_{2},\sigma^{2})$, respectively. Given the $n = 6$ observations of $X$,


```r
x = c(70, 82, 78, 74, 94, 82)
n = length(x)
```

and the $m = 8$ observations of $Y$,


```r
y = c(64, 72, 60, 76, 72, 80, 84, 68)
m = length(y)
```

we will test $H_{0}: \mu_{1} = \mu_{2}$ versus $H_{1}: \mu_{1} > \mu_{2}$.

First, note that we can calculate the sample means and standard deviations.


```r
x_bar = mean(x)
s_x   = sd(x)
y_bar = mean(y)
s_y   = sd(y)
```

We can then calculate the pooled standard deviation.

\[
s_{p} = \sqrt{\frac{(n-1)s_{x}^{2}+(m-1)s_{y}^{2}}{n+m-2}}
\]


```r
s_p = sqrt(((n - 1) * s_x ^ 2 + (m - 1) * s_y ^ 2) / (n + m - 2))
```

Thus, the relevant $t$ test statistic is given by

\[
t = \frac{(\bar{x}-\bar{y})-\mu_{0}}{s_{p}\sqrt{\frac{1}{n}+\frac{1}{m}}}.
\]


```r
t = ((x_bar - y_bar) - 0) / (s_p * sqrt(1 / n + 1 / m))
t
```

```
## [1] 1.823369
```

Note that $t \sim t_{n + m - 2} = t_{12}$, so we can calculate the p-value, which is

\[
P(t_{12} > 1.8233692).
\]


```r
1 - pt(t, df = n + m - 2)
```

```
## [1] 0.04661961
```

But, then again, we could have simply performed this test in one line of `R`.


```r
t.test(x, y, alternative = c("greater"), var.equal = TRUE)
```

```
## 
## 	Two Sample t-test
## 
## data:  x and y
## t = 1.8234, df = 12, p-value = 0.04662
## alternative hypothesis: true difference in means is greater than 0
## 95 percent confidence interval:
##  0.1802451       Inf
## sample estimates:
## mean of x mean of y 
##        80        72
```

Recall that a two-sample $t$-test can be done with or without an equal variance assumption. Here `var.equal = TRUE` tells `R` we would like to perform the test under the equal variance assumption.

Above we carried out the analysis using two vectors `x` and `y`. In general, we will have a preference for using data frames.


```r
t_test_data = data.frame(values = c(x, y),
                         group  = c(rep("A", length(x)), rep("B", length(y))))
```

We now have the data stored in a single variables (`values`) and have created a second variable (`group`) which indicates which "sample" the value belongs to.


```r
t_test_data
```

```
##    values group
## 1      70     A
## 2      82     A
## 3      78     A
## 4      74     A
## 5      94     A
## 6      82     A
## 7      64     B
## 8      72     B
## 9      60     B
## 10     76     B
## 11     72     B
## 12     80     B
## 13     84     B
## 14     68     B
```

Now to perform the test, we still use the `t.test()` function but with the `~` syntax and a `data` argument.


```r
t.test(values ~ group, data = t_test_data,
       alternative = c("greater"), var.equal = TRUE)
```

```
## 
## 	Two Sample t-test
## 
## data:  values by group
## t = 1.8234, df = 12, p-value = 0.04662
## alternative hypothesis: true difference in means is greater than 0
## 95 percent confidence interval:
##  0.1802451       Inf
## sample estimates:
## mean in group A mean in group B 
##              80              72
```

## Simulation

Simulation and model fitting are related but opposite processes.

- In **simulation**, the *data generating process* is known. We will know the form of the model as well as the value of each of the parameters. In particular, we will often control the distribution and parameters which define the randomness, or noise in the data.
- In **model fitting**, the *data* is known. We will then assume a certain form of model and find the best possible values of the parameters given the observed data. Essentially we are seeking to uncover the truth. Often we will attempt to fit many models, and we will learn metrics to assess which model fits best.

![Simulation vs Modeling](images/simulation.png)

Often we will simulate data according to a process we decide, then use a modeling method seen in class. We can then verify how well the method works, since we know the data generating process.

One of the biggest strengths of `R` is its ability to carry out simulations using built-in functions for generating random samples from certain distributions. We'll look at two very simple examples here, however simulation will be a topic we revisit several times throughout the course.

### Paired Differences

Consider the model:

\[
\begin{split}
X_{11}, X_{12}, \ldots, X_{1n} \sim N(\mu_1,\sigma^2)\\
X_{21}, X_{22}, \ldots, X_{2n} \sim N(\mu_2,\sigma^2)
\end{split}
\]

Assume that $\mu_1 = 6$, $\mu_2 = 5$, $\sigma^2 = 4$ and $n = 25$.

Let

\[
\begin{aligned}
\bar{X}_1 &= \displaystyle\frac{1}{n}\sum_{i=1}^{n}X_{1i}\\
\bar{X}_2 &= \displaystyle\frac{1}{n}\sum_{i=1}^{n}X_{2i}\\
D &= \bar{X}_1 - \bar{X}_2.
\end{aligned}
\]

Suppose we would like to calculate $P(0 < D < 2)$. First we will need to obtain the distribution of $D$.

Recall,

\[
\bar{X}_1 \sim N\left(\mu_1,\frac{\sigma^2}{n}\right)
\]

and

\[
\bar{X}_2 \sim N\left(\mu_2,\frac{\sigma^2}{n}\right).
\]

Then, 

\[
D = \bar{X}_1 - \bar{X}_2 \sim N\left(\mu_1-\mu_2, \frac{\sigma^2}{n} + \frac{\sigma^2}{n}\right) = N\left(6-5, \frac{4}{25} + \frac{4}{25}\right).
\]

So, 

\[
D \sim N(\mu = 1, \sigma^2 = 0.32).
\]

Thus,

\[
P(0 < D < 2) = P(D < 2) - P(D < 0).
\]

This can then be calculated using `R` without a need to first standardize, or use a table.


```r
pnorm(2, mean = 1, sd = sqrt(0.32)) - pnorm(0, mean = 1, sd = sqrt(0.32))
```

```
## [1] 0.9229001
```

An alternative approach, would be to **simulate** a large number of observations of $D$ then use the **empirical distribution** to calculate the probability.

Our strategy will be to repeatedly:

- Generate a sample of 25 random observations from $N(\mu_1 = 6,\sigma^2 = 4)$. Call the mean of this sample $\bar{x}_{1s}$.
- Generate a sample of 25 random observations from $N(\mu_1 = 5,\sigma^2 = 4)$. Call the mean of this sample $\bar{x}_{2s}$.
- Calculate the differences of the means, $d_s = \bar{x}_{1s} - \bar{x}_{2s}$.

We will repeat the process a large number of times. Then we will use the distribution of the simulated observations of $d_s$ as an estimate for the true distribution of $D$.


```r
set.seed(42)
num_samples = 10000
differences = rep(0, num_samples)
```

Before starting our `for` loop to perform the operation, we set a seed for reproducibility, create and set a variable `num_samples` which will define the number of repetitions, and lastly create a variables `differences` which will store the simulate values, $d_s$.

By using `set.seed()` we can reproduce the random results of `rnorm()` each time starting from that line. 


```r
for (s in 1:num_samples) {
  x1 = rnorm(n = 25, mean = 6, sd = 2)
  x2 = rnorm(n = 25, mean = 5, sd = 2)
  differences[s] = mean(x1) - mean(x2)
}
```

To estimate $P(0 < D < 2)$ we will find the proportion of values of $d_s$ (among the 10^{4} values of $d_s$ generated) that are between 0 and 2.


```r
mean(0 < differences & differences < 2)
```

```
## [1] 0.9222
```

Recall that above we derived the distribution of $D$ to be $N(\mu = 1, \sigma^2 = 0.32)$

If we look at a histogram of the differences, we find that it looks very much like a normal distribution.


```r
hist(differences, breaks = 20, 
     main   = "Empirical Distribution of D",
     xlab   = "Simulated Values of D",
     col    = "dodgerblue",
     border = "darkorange")
```

![](02-r-intro_files/figure-latex/unnamed-chunk-124-1.pdf)<!-- --> 

Also the sample mean and variance are very close to to what we would expect.


```r
mean(differences)
```

```
## [1] 1.001423
```

```r
var(differences)
```

```
## [1] 0.3230183
```

We could have also accomplished this task with a single line of more "idiomatic" `R`.


```r
set.seed(42)
diffs = replicate(10000, mean(rnorm(25, 6, 2)) - mean(rnorm(25, 5, 2)))
```

Use `?replicate` to take a look at the documentation for the `replicate` function and see if you can understand how this line performs the same operations that our `for` loop above executed.


```r
mean(differences == diffs)
```

```
## [1] 1
```

We see that by setting the same seed for the randomization, we actually obtain identical results!

### Distribution of a Sample Mean

For another example of simulation, we will simulate observations from a Poisson distribution, and examine the empirical distribution of the sample mean of these observations.

Recall, if

\[
X \sim Pois(\mu)
\]

then

\[
E[X] = \mu
\]

and

\[
Var[X] = \mu.
\]

Also, recall that for a random variable $X$ with finite mean $\mu$ and finite variance $\sigma^2$, the central limit theorem tells us that the mean, $\bar{X}$ of a random sample of size $n$ is approximately normal for *large* values of $n$. Specifically, as $n \to \infty$,

\[
\bar{X} \overset{d}{\to} N\left(\mu, \frac{\sigma^2}{n}\right).
\]

The following verifies this result for a Poisson distribution with $\mu = 10$ and a sample size of $n = 50$.


```r
set.seed(1337)
mu          = 10
sample_size = 50
samples     = 100000
x_bars      = rep(0, samples)
```


```r
for(i in 1:samples){
  x_bars[i] = mean(rpois(sample_size, lambda = mu))
}
```


```r
x_bar_hist = hist(x_bars, breaks = 50, 
                  main = "Histogram of Sample Means",
                  xlab = "Sample Means")
```

![](02-r-intro_files/figure-latex/unnamed-chunk-130-1.pdf)<!-- --> 

Now we will compare sample statistics from the empirical distribution with their known values based on the parent distribution.


```r
c(mean(x_bars), mu)
```

```
## [1] 10.00008 10.00000
```


```r
c(var(x_bars), mu / sample_size)
```

```
## [1] 0.1989732 0.2000000
```


```r
c(sd(x_bars), sqrt(mu) / sqrt(sample_size))
```

```
## [1] 0.4460641 0.4472136
```

And here, we will calculate the proportion of sample means that are within 2 standard deviations of the population mean.


```r
mean(x_bars > mu - 2 * sqrt(mu) / sqrt(sample_size) &
     x_bars < mu + 2 * sqrt(mu) / sqrt(sample_size))
```

```
## [1] 0.95429
```

This last histogram uses a bit of a trick to approximately shade the bars that are within two standard deviations of the mean.)


```r
shading = ifelse(x_bar_hist$breaks > mu - 2 * sqrt(mu) / sqrt(sample_size) & 
                   x_bar_hist$breaks < mu + 2 * sqrt(mu) / sqrt(sample_size),
                  "darkorange", "dodgerblue")

x_bar_hist = hist(x_bars, breaks = 50, col = shading,
                  main = "Histogram of Sample Means, Two Standard Deviations",
                  xlab = "Sample Means")
```

![](02-r-intro_files/figure-latex/unnamed-chunk-135-1.pdf)<!-- --> 
