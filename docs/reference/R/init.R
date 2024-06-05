
# jupyter lab
# reticulate::py_last_error()

devtools::install_github("Sendrowski/PhaseGen")

sink(file = stderr(), type = "message")

library(phasegen)

# install_phasegen()

setwd("~/PycharmProjects/PhaseGen/")

pg <- load_phasegen()

pg$logger$setLevel("DEBUG")
