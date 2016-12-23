#!/usr/bin/env Rscript

bookdown::render_book("chapters/index.Rmd", "bookdown::gitbook")
bookdown::render_book("chapters/index.Rmd", "bookdown::pdf_book")
