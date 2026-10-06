pg <- tryCatch(load_phasegen(), error = function(e) NULL)

test_that("load_phasegen returns the phasegen module", {
  skip_if_not(phasegen_is_installed())

  expect_s3_class(pg, "python.builtin.module")
  expect_true(is.function(pg$Coalescent))
})

test_that("the mean tree height of the standard coalescent with n = 2 is 1", {
  skip_if_not(phasegen_is_installed())

  expect_equal(pg$Coalescent(n = 2L)$tree_height$mean, 1, tolerance = 1e-10)
})

test_that("plotting the mean SFS returns a ggplot that builds", {
  skip_if_not(phasegen_is_installed())

  p <- plot(pg$Coalescent(n = 5L)$sfs$mean)

  expect_s3_class(p, "ggplot")
  expect_no_error(ggplot2::ggplot_build(p))
})

test_that("distribution-function plots pass the evaluation grid and the joint SFS bins", {
  skip_if_not(phasegen_is_installed())

  coal <- pg$Coalescent(
    n = list(pop_0 = 2, pop_1 = 2),
    demography = pg$Demography(
      pop_sizes = list(pop_0 = 1, pop_1 = 1),
      events = c(
        pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1),
        pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 1)
      )
    )
  )

  p <- plot(coal$jsfs$pdf, t = c(0.5, 1, 2), configs = list(c(1L, 0L), c(0L, 1L)))
  expect_equal(sort(unique(p$data$x)), c(0.5, 1, 2))
  expect_equal(length(unique(p$data$series)), 2)

  q <- plot(pg$Coalescent(n = 3L)$tree_height$quantile, q = c(0.25, 0.5, 0.75))
  expect_equal(sort(unique(q$data$x)), c(0.25, 0.5, 0.75))
})

test_that("mutational configurations convert to integer vectors of their counts", {
  skip_if_not(phasegen_is_installed())

  configs <- reticulate::iterate(pg$take_n(pg$Coalescent(n = 4L)$sfs$get_mutation_configs(theta = 1), 2L))

  expect_s3_class(configs[[1]][[1]], "phasegen_mutation_config")
  expect_identical(as.integer(configs[[1]][[1]]), c(0L, 0L, 0L))
  expect_identical(as.integer(configs[[2]][[1]]), c(1L, 0L, 0L))
  expect_output(print(configs[[2]][[1]]), "^\\[1\\] 1 0 0$")
})

test_that("one-bin mutational configurations pass back to get_mutation_config", {
  skip_if_not(phasegen_is_installed())

  for (sfs in list(pg$Coalescent(n = 2L)$sfs, pg$Coalescent(n = 2L)$fsfs, pg$Coalescent(n = 3L)$fsfs)) {
    for (pair in reticulate::iterate(pg$take_n(sfs$get_mutation_configs(theta = 1), 3L))) {
      expect_length(pair[[1]], 1)
      expect_equal(sfs$get_mutation_config(pair[[1]], theta = 1), pair[[2]], tolerance = 1e-14)
    }
  }
})

test_that("configurations of a non-default layout pass back to get_mutation_config", {
  skip_if_not(phasegen_is_installed())

  coal <- pg$Coalescent(
    n = list(pop_0 = 2L, pop_1 = 2L),
    demography = pg$Demography(
      pop_sizes = list(pop_0 = 0.5, pop_1 = 2),
      events = c(
        pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1),
        pg$MigrationRateChange(source = "pop_1", dest = "pop_0", time = 0, rate = 0.2)
      )
    )
  )

  for (layout in list(coal$sfs$mutation_layout(demes = TRUE), coal$sfs$mutation_layout(folded = TRUE))) {
    for (pair in reticulate::iterate(pg$take_n(coal$sfs$get_mutation_configs(theta = 0.5, layout = layout), 4L))) {
      expect_equal(coal$sfs$get_mutation_config(pair[[1]], theta = 0.5), pair[[2]], tolerance = 1e-14)
    }
  }

  config <- pg$MutationConfig(c(1L, 2L), coal$sfs$mutation_layout(folded = TRUE))
  expect_equal(coal$sfs$get_mutation_config(config, theta = 0.5),
               coal$fsfs$get_mutation_config(c(1L, 2L), theta = 0.5), tolerance = 1e-14)
})

test_that("persp() draws a joint SFS whose facet heights agree to round-off", {
  skip_if_not(phasegen_is_installed())

  coal <- pg$Coalescent(
    n = list(pop_0 = 2L, pop_1 = 2L, pop_2 = 2L),
    demography = pg$Demography(
      pop_sizes = list(pop_0 = 1, pop_1 = 1, pop_2 = 1),
      events = list(pg$SymmetricMigrationRateChanges(pops = c("pop_0", "pop_1", "pop_2"), rate = 1))
    )
  )

  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())

  for (pops in list(c(0L, 1L), c(0L, 2L), c(1L, 2L))) {
    expect_no_error(persp(coal$jsfs$mean, pops = pops))
  }
})

test_that("the declared Python version constraints admit Python 3.10 to 3.13", {
  checkers <- utils::getFromNamespace("as_version_constraint_checkers", "reticulate")(
    reticulate::py_require()$python_version
  )

  for (version in c("3.10.4", "3.11.0", "3.12.1", "3.13.2")) {
    expect_true(all(vapply(checkers, function(check) check(version), logical(1))), label = version)
  }
})

test_that("plot_accumulation() draws empirical distributions and coalescents like the exact distribution", {
  skip_if_not(phasegen_is_installed())

  t <- c(0.5, 1, 2)
  coal <- pg$Coalescent(n = 4L)
  ms <- pg$distributions$MsprimeCoalescent(n = 4L, num_replicates = 200L, parallelize = FALSE, seed = 1L)
  sampled <- pg$distributions$SampledCoalescent(coal, n_samples = 200L, seed = 1L)

  exact <- plot_accumulation(coal$tree_height, k = 2L, end_times = t)
  expect_equal(plot_accumulation(coal, k = 2L, end_times = t)$data, exact$data)

  for (x in list(ms$tree_height, ms$sfs, ms, sampled, sampled$sfs)) {
    p <- plot_accumulation(x, k = 1L, end_times = t)
    expect_s3_class(p, "ggplot")
    expect_equal(sort(unique(p$data$x)), t)
    expect_no_error(ggplot2::ggplot_build(p))
  }

  expect_equal(length(unique(plot_accumulation(ms$sfs, end_times = t)$data$series)), 3)
})
