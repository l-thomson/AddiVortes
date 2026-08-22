drive_small <- function(x, y, seed) {
  sampler <- thiessen_sampler(x, y, small_control(), seed = seed)
  sampler$step(10)
  for (draw in seq_len(20)) {
    sampler$step(1)
    sampler$keep()
  }
  sampler
}

test_that("a driven fit matches thiessen() bit for bit", {
  fixture <- core_fixture()

  through_fit <- thiessen(fixture$x, fixture$y, small_control(), seed = 1)
  through_sampler <- drive_small(fixture$x, fixture$y, seed = 1)$finish()

  expect_identical(
    predict(through_sampler, type = "draws"),
    predict(through_fit, type = "draws")
  )
  expect_identical(sigma(through_sampler), sigma(through_fit))
})

test_that("thinning is the caller's loop", {
  fixture <- core_fixture()
  control <- thiessen_control(
    tessellations = 8,
    general_params = general_params(burn_in = 10, draws = 20, thinning = 3)
  )
  thinned <- thiessen(fixture$x, fixture$y, control, seed = 1)

  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)
  sampler$step(10)
  for (draw in seq_len(20)) {
    sampler$step(3)
    sampler$keep()
  }
  driven <- sampler$finish()

  expect_identical(
    predict(driven, type = "draws"),
    predict(thinned, type = "draws")
  )
})

test_that("finish returns the ordinary fit object", {
  fixture <- small_fixture()
  fit <- drive_small(fixture$x, fixture$y, seed = 1)$finish()

  expect_s3_class(fit, "thiessen")
  expect_identical(fit$model, "gaussian")
  expect_identical(fit$n_draws, 20L)
  expect_identical(fit$control$mean_params$tessellations, 8L)
  expect_length(predict(fit), nrow(fixture$x))
})

test_that("n_kept counts the kept draws", {
  fixture <- small_fixture()
  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)

  expect_identical(sampler$n_kept(), 0L)
  sampler$step(2)
  sampler$keep()
  expect_identical(sampler$n_kept(), 1L)
})

test_that("fitted values and noise variances have one value per row", {
  fixture <- small_fixture()
  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)
  sampler$step(2)

  expect_length(sampler$fitted_values(), nrow(fixture$x))
  variances <- sampler$noise_variances()
  expect_length(variances, nrow(fixture$x))
  expect_true(all(variances > 0))
})

test_that("a replaced response conditions the next sweep", {
  fixture <- core_fixture()
  unchanged <- drive_small(fixture$x, fixture$y, seed = 1)$finish()

  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)
  sampler$step(10)
  sampler$set_response(fixture$y + 0.5 * sin(6 * fixture$x[, 1]))
  for (draw in seq_len(20)) {
    sampler$step(1)
    sampler$keep()
  }
  swapped <- sampler$finish()

  expect_false(identical(
    predict(swapped, type = "draws"),
    predict(unchanged, type = "draws")
  ))
})

test_that("a response outside the training range is legitimate", {
  fixture <- small_fixture()
  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)
  sampler$step(2)

  sampler$set_response(fixture$y + 100)
  sampler$step(2)

  expect_length(sampler$fitted_values(), nrow(fixture$x))
})

test_that("rejections keep their reason", {
  fixture <- small_fixture()
  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)

  expect_error(sampler$set_response(fixture$y[-1]), class = "thiessen_error")
  bad <- fixture$y
  bad[1] <- NA_real_
  expect_error(sampler$set_response(bad), class = "thiessen_error")
  expect_error(sampler$step(-1), class = "thiessen_error")
  expect_error(sampler$finish(), "no draws were kept",
               class = "thiessen_error")
})

test_that("every call after finish errors", {
  fixture <- small_fixture()
  sampler <- drive_small(fixture$x, fixture$y, seed = 1)
  invisible(sampler$finish())

  expect_error(sampler$step(1), "finished", class = "thiessen_error")
  expect_error(sampler$keep(), "finished", class = "thiessen_error")
  expect_error(sampler$fitted_values(), "finished", class = "thiessen_error")
  expect_error(sampler$finish(), "finished", class = "thiessen_error")
  expect_output(print(sampler), "finished")
})

test_that("the constructor validates its arguments", {
  fixture <- small_fixture()

  expect_error(
    thiessen_sampler(fixture$x, fixture$y, control = list()),
    class = "thiessen_error"
  )
  expect_error(
    thiessen_sampler(fixture$x, fixture$y[-1], small_control(), seed = 1),
    class = "thiessen_error"
  )
})

test_that("printing reports the kept draws", {
  fixture <- small_fixture()
  sampler <- thiessen_sampler(fixture$x, fixture$y, small_control(), seed = 1)

  expect_output(print(sampler), "0 draw")
})
