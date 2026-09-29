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
