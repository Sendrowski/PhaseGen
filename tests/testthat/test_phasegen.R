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
