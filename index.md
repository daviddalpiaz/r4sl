--- 
title: "`R` for Statistical Learning"
author: "[David Dalpiaz](https://daviddalpiaz.com/)"
date: "2017-08-25"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib]
biblio-style: apalike
link-citations: yes
github-repo: daviddalpiaz/r4sl
url: 'http\://daviddalpiaz.github.io/r4sl/'
description: ""
favicon: "favicon.ico"
---

# Introduction {-}

Welcome to `R` for Statistical Learning!


## About This Book

This book will serve as a supplement to [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/) for [STAT 430 - Basics of Statistical Learning](https://go.illinois.edu/stat430) at the [University of Illinois at Urbana-Champaign](http://illinois.edu/).

Chapters will come in roughly three flavors:

- **Notes** that discuss mathematics in greater detail. 
- **Tutorials** that illustrate the use of `R` for statistical learning.
- **Analyses** that show end-to-end analysis of a particular dataset.

The end of each chapter will contain:

- Annotated links to additional information and resources.
- A link to the RMarkdown file that generates the chapter.


## Caveat Emptor

**This "book" is under active development.** Chapters will be added as we move through the course in Spring 2017. Sometimes chapters will be more in the style of course notes than a fully narrative text.

When possible, it would be best to always access the text online to be sure you are using the most up-to-date version. Also, the html version provides additional features such as changing text size, font, and colors. If you are in need of a local copy, a [**pdf version** is continuously maintained](http://daviddalpiaz.github.io/r4sl/r4sl.pdf).

Since this book is under active development you may encounter errors ranging from typos, to broken code, to poorly explained topics. If you do, please let us know! Simply send an email and we will make the changes as soon as possible. (`dalpiaz2 AT illinois DOT edu`) Or, if you know RMarkdown and are familiar with GitHub, [make a pull request and fix an issue yourself!](https://github.com/daviddalpiaz/r4sl) This process is partially automated by the edit button in the top-left corner of the html version. If your suggestion or fix becomes part of the book, you will be added to the list at the end of this chapter. We'll also link to your GitHub account, or personal website upon request.


## Conventions

This text uses MathJax to render mathematical notation for the web. Occasionally, but rarely, a JavaScript error will prevent MathJax from rendering correctly. In this case, you will see the "code" instead of the expected mathematical equations. From experience, this is almost always fixed by simply refreshing the page. You'll also notice that if you right-click any equation you can obtain the MathML Code (for copying into Microsoft Word) or the TeX command used to generate the equation.

\[
a^2 + b^2 = c^2
\]

`R` code will be typeset using a `monospace` font which is syntax highlighted.


```r
a = 3
b = 4
sqrt(a ^ 2 + b ^ 2)
```

`R` output lines, which would appear in the console will begin with `##`. They will generally not be syntax highlighted.


```
## [1] 5
```

Often the symbol $\triangleq$ will be used to mean "is defined to be."

We use the value $p$ to mean the number of **p**redictors.


## Acknowledgements

Your name could be here! Suggest an edit! Correct a typo! If you submit a correction and would like to be listed below, please provide your name as you would like it to appear, as well as a link to a GitHub, LinkedIn, or personal website.

- [James Balamuta](http://www.thecoatlessprofessor.com/), Summer 2016 - ??? 
- Korawat Tanwisuth, Spring 2017
- [Yiming Gao](https://www.linkedin.com/in/yiming-gao), Spring 2017
- [Binxiang Ni](https://github.com/binxiangni), Summer 2017


## License

![This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).](images/cc.png)
