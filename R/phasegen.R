
# vector of required packages
required_packages <- c("reticulate")

# install required R packages
for(package in required_packages){
  if(!package %in% installed.packages()[,"Package"]){
    install.packages(package)
  }
}

#' Check if the `phasegen` Python module is installed
#'
#' This function uses the reticulate package to verify if the `phasegen` Python
#' module is currently installed. 
#'
#' @return Logical `TRUE` if the `phasegen` Python module is installed, otherwise `FALSE`.
#'
#' @examples
#' \dontrun{
#' is_installed()  # Returns TRUE or FALSE based on the installation status of phasegen
#' }
#' 
#' @export
phasegen_is_installed <- function() {

  # An unbound session reports FALSE without touching Python, leaving the interpreter
  # for the declared requirements to select at the version they ask for
  if (!reticulate::py_available(initialize = FALSE)) {
    return(FALSE)
  }

  # Check if phasegen is installed
  return(reticulate::py_module_available("phasegen"))
}


# Requirement string for the Python distribution, carrying a pinned version where one is given
py_requirement <- function(version = NULL) {

  spec <- "phasegen"

  if (!is.null(version)) {
    spec <- paste0(spec, "==", version)
  }

  spec
}


.onLoad <- function(libname, pkgname) {
  reticulate::py_require(py_requirement(), python_version = "3.11")
}


#' Declare the `phasegen` Python module requirement
#'
#' Loading the package declares `phasegen`. This function declares a pinned version. The requirement is resolved when Python is first initialised, at which
#' point reticulate provisions an environment satisfying it.
#'
#' @param version A character string specifying the version of the `phasegen` module
#'        to require. Default is `NULL` which resolves to the latest version.
#' @param force Logical, has no effect. Default is `FALSE`.
#' @param silent Logical, if `TRUE` it will suppress the message naming the declared
#'        requirement. Default is `FALSE`.
#' @param python_version A character string specifying the Python version reticulate
#'        should provision the environment with. Default is `'3.11'`.
#'
#' @return Invisible `NULL`.
#'
#' @examples
#' \dontrun{
#' install_phasegen()  # Requires the latest version of phasegen
#' }
#'
#' @export
install_phasegen <- function(version = NULL, force = FALSE, silent = FALSE, python_version = '3.11') {

  if (force) {
    warning("'force' has no effect.", call. = FALSE)
  }

  spec <- py_requirement(version)

  reticulate::py_require(spec, python_version = python_version)

  if (!silent) {
    message("Declared Python requirement '", spec, "' on Python ", python_version, ".")
  }

  invisible(NULL)
}

#' Load the phasegen library
#'
#' This function imports the Python package 'phasegen' using the reticulate package
#' and returns a reference to it, optionally installing it first when `install = TRUE`.
#'
#' @param install A logical. If TRUE, the function will attempt to run install_phasegen().
#'
#' @return A reference to the 'phasegen' Python library loaded through reticulate.
#'         This reference can be used to access 'phasegen' functionalities.
#'
#' @examples
#' \dontrun{
#' load_phasegen(install = TRUE)
#' # now you can use phasegen functionalities as per its API
#' }
#'
#' @seealso \link[reticulate]{import} for importing Python modules in R.
#'
#' @export
load_phasegen <- function(install = FALSE) {
  
  # install if install flag is true
  if (install) {
    install_phasegen(silent = TRUE)
  }
  
  forward_python_output()

  pg <- reticulate::import("phasegen")
}


# In a Jupyter kernel, write Python's standard output and error through R's output and message streams, which the kernel
# captures, so log messages and progress bars reach the cell output. Runs before phasegen is imported, as its log
# handler binds the error stream at import.
forward_python_output <- function() {

  if (!isTRUE(getOption("jupyter.in_kernel"))) {
    return(invisible(NULL))
  }

  # the kernel ends every message with a line break, so the error stream is passed on as complete lines without one
  streams <- reticulate::py_run_string("
class RStream:
    def __init__(self, write, lines=False):
        self._write = write
        self._lines = lines
        self._buffer = ''

    def write(self, text):
        if not self._lines:
            self._write(text)
            return len(text)

        *complete, self._buffer = (self._buffer + text).split('\\n')
        for line in complete:
            self._write(line)

        return len(text)

    def flush(self):
        if self._lines and self._buffer:
            self._write(self._buffer)
            self._buffer = ''
", local = TRUE, convert = FALSE)

  sys <- reticulate::import("sys", convert = FALSE)
  sys$stdout <- streams$RStream(function(text) cat(reticulate::py_to_r(text)))
  sys$stderr <- streams$RStream(
    function(line) message(reticulate::py_to_r(line), appendLF = FALSE),
    lines = TRUE
  )

  invisible(NULL)
}


# R-native plotting of phasegen objects: ggplot2 for 2D figures, graphics::persp() for surfaces. The methods dispatch
# on the class vectors reticulate assigns to Python objects, so plot(coal$sfs$mean) and persp(joint$pdf) work directly.


#' @importFrom ggplot2 .data
NULL


# matplotlib's default colour cycle, so that the R figures colour series as the Python ones do
tab10 <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
           "#17becf")


# colour vector of length n cycling through tab10
series_colours <- function(n) {
  rep_len(tab10, n)
}


# darken colours by a factor, used for bar outlines
darken <- function(colours, amount = 0.75) {
  rgb <- grDevices::col2rgb(colours) * amount
  grDevices::rgb(rgb[1, ], rgb[2, ], rgb[3, ], maxColorValue = 255)
}


# Python helpers for data that does not convert to R directly: the epochs of a demography (an object array with
# tuple-keyed migration dictionaries) and the bootstrap table of an inference (which carries an object column)
py_helpers_env <- new.env(parent = emptyenv())

py_helpers <- function() {

  if (is.null(py_helpers_env$helpers)) {
    # the code runs with its own local namespace, which the functions do not see, so each imports what it uses
    py_helpers_env$helpers <- reticulate::py_run_string("
def rates(d, t, kind):
    import numpy as np
    epochs = d.get_epochs(np.asarray(t, dtype=float))
    pops = list(d.pop_names)
    if kind == 'pop_sizes':
        keys, names = pops, pops
        values = [[e.pop_sizes[p] for p in keys] for e in epochs]
    else:
        keys = [(p, q) for p in pops for q in pops if p != q]
        names = [f'{p}->{q}' for p, q in keys]
        values = [[e.migration_rates[k] for k in keys] for e in epochs]
    if not keys:
        return [], None
    return names, np.array(values, dtype=float)

def bootstraps(inf):
    df = inf.bootstraps[inf.param_names]
    if len(df) == 0:
        return list(df.columns), None
    return list(df.columns), df.to_numpy(dtype=float)

def bootstrap_demographies(inf):
    return [inf.get_coal(**row.to_dict()).demography for _, row in inf.bootstraps[inf.param_names].iterrows()]

def samples_shape(d):
    import numpy as np
    return list(np.shape(d.samples))

def type_name(obj):
    return type(obj).__name__
", local = TRUE)
  }

  py_helpers_env$helpers
}


# plotting grid defaults from phasegen's Settings
plot_settings <- function() {
  settings <- reticulate::import("phasegen")$Settings

  list(q_end = settings$plot_endpoint_quantile, n_grid = as.integer(settings$plot_n_grid))
}


# theme of all 2D figures: the current ggplot2 theme (see ggplot2::theme_set()) without grid lines and with a centred
# title
theme_phasegen <- function() {
  ggplot2::theme_get() +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(hjust = 0.5)
    )
}


# Corner of the panel, in normalised panel coordinates, whose neighbourhood holds the fewest points of the curves
legend_corner <- function(data, corners = list(c(1, 1), c(0, 1), c(1, 0), c(0, 0))) {

  rescale <- function(v) {
    r <- range(v, finite = TRUE)
    if (diff(r) == 0) rep(0.5, length(v)) else (v - r[1]) / diff(r)
  }

  x <- rescale(data$x)
  y <- rescale(data$y)

  counts <- vapply(corners, function(k) sum(abs(x - k[1]) < 0.4 & abs(y - k[2]) < 0.4, na.rm = TRUE), numeric(1))

  corners[[which.min(counts)]]
}


# Legend inside the panel, inset from the given corner
inside_legend <- function(corner) {
  ggplot2::theme(
    legend.position = "inside",
    legend.position.inside = abs(corner - 0.03),
    legend.justification = corner,
    legend.background = ggplot2::element_rect(fill = grDevices::adjustcolor("white", 0.8), colour = "grey80")
  )
}


# Line plot of one or more curves. `data` has columns x, y and series, where series is a factor or is converted to one
# in order of appearance. A legend is drawn for more than one series. Layers in `under` are drawn beneath the curves.
# The x-axis has no margins, and a given `ylim` is drawn without margins either.
line_plot <- function(data, xlab, ylab, title, legend = NULL, step = FALSE, ylim = NULL, under = NULL) {

  if (!is.factor(data$series)) {
    data$series <- factor(data$series, levels = unique(data$series))
  }

  n_series <- nlevels(data$series)

  if (is.null(data$linewidth)) {
    data$linewidth <- 0.5
  }

  if (is.null(data$alpha)) {
    data$alpha <- 1
  }

  style <- ggplot2::aes(linewidth = .data$linewidth, alpha = .data$alpha)
  geom <- if (step) ggplot2::geom_step(style, direction = "hv") else ggplot2::geom_line(style)

  p <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y, colour = .data$series)) +
    under +
    geom +
    ggplot2::scale_colour_manual(values = series_colours(n_series), drop = FALSE,
                                 guide = if (n_series > 1) "legend" else "none") +
    ggplot2::scale_linewidth_identity() +
    ggplot2::scale_alpha_identity() +
    ggplot2::labs(x = xlab, y = ylab, title = title, colour = legend) +
    theme_phasegen()

  if (n_series > 1) {
    p <- p + inside_legend(legend_corner(data))
  }

  if (is.null(ylim)) {
    p + ggplot2::scale_x_continuous(expand = c(0, 0))
  } else {
    p + ggplot2::coord_cartesian(ylim = ylim, expand = FALSE)
  }
}


# Surface of z over the grid x by y (z has dimension length(x) by length(y)) drawn with persp(), each facet coloured
# by its mean height on the viridis palette
draw_surface <- function(x, y, z, xlab, ylab, zlab, title, zlim = NULL, theta = 30, phi = 30, border = NULL,
                         n_colours = 100, ...) {

  nx <- nrow(z)
  ny <- ncol(z)

  facets <- (z[-1, -1] + z[-1, -ny] + z[-nx, -1] + z[-nx, -ny]) / 4

  # the palette spans the facet heights, or a given zlim
  colour_range <- if (is.null(zlim)) range(facets, finite = TRUE) else zlim

  if (is.null(zlim)) {
    zlim <- range(z[is.finite(z)])
  }

  if (diff(zlim) == 0) {
    zlim <- zlim + c(-0.5, 0.5)
  }

  if (diff(colour_range) == 0) {
    colour_range <- colour_range + c(-0.5, 0.5)
  }

  facets <- pmin(pmax(facets, colour_range[1]), colour_range[2])

  breaks <- seq(colour_range[1], colour_range[2], length.out = n_colours + 1)
  colours <- grDevices::hcl.colors(n_colours, "viridis")[cut(facets, breaks, include.lowest = TRUE, labels = FALSE)]

  # facets are outlined on coarse grids only, as outlines on a fine grid cover the surface
  if (is.null(border)) {
    border <- if (max(nx, ny) > 25) NA else "grey20"
  }

  # persp() places the axis titles close to the tick labels: small labels and a leading line break, which moves each
  # title one line outwards, keep them apart
  pad <- function(label) if (nzchar(label)) paste0("\n", label) else label
  args <- list(...)

  if (is.null(args$cex.axis)) {
    args$cex.axis <- 0.6
  }

  if (is.null(args$cex.lab)) {
    args$cex.lab <- 0.8
  }

  # the default margins leave the projected box small within the figure
  op <- graphics::par(mar = c(1.8, 0.8, 1.8, 0.8))
  on.exit(graphics::par(op))

  transform <- do.call(graphics::persp, c(
    list(x = x, y = y, z = z, zlim = zlim, col = colours, border = border, theta = theta, phi = phi,
         ticktype = "detailed", xlab = pad(xlab), ylab = pad(ylab), zlab = pad(zlab), main = title),
    args
  ))

  invisible(transform)
}


# ---- univariate distribution functions ------------------------------------------------------------------------------


# The flavour of a univariate distribution function, which determines the plotting grid and the labels
function_flavour <- function(x) {

  if (inherits(x, "phasegen.distributions.base._LSTFunction")) {
    return("lst")
  }

  if (any(startsWith(class(x), "phasegen.distributions.base._Grid"))) {
    return("grid")
  }

  if (any(startsWith(class(x), "phasegen.distributions.empirical._Empirical"))) {
    return("empirical")
  }

  if (inherits(x, "phasegen.distributions.spectra._SFSAggregateFunction")) {
    return("sfs")
  }

  if (inherits(x$`_distribution`, "phasegen.distributions.phase_type.PhaseTypeDistribution") &&
      !inherits(x$`_distribution`, "phasegen.distributions.spectra.JointSFSDistribution")) {
    return("reward")
  }

  stop("No R plotting method for '", class(x)[1], "'. Use the Python method x$plot() instead.", call. = FALSE)
}


# Curves of a univariate distribution function on the grid phasegen's own plot uses: a data frame with columns x, y
# and series, together with the default labels and title
function_curves <- function(x, kind, n_points = NULL, bins = NULL) {

  settings <- plot_settings()
  n_points <- if (is.null(n_points)) settings$n_grid else as.integer(n_points)
  q_end <- settings$q_end
  d <- x$`_distribution`
  flavour <- function_flavour(x)

  ylab <- switch(kind, pdf = "f(x)", cdf = "F(x)", quantile = "quantile")
  q_grid <- seq(1 - q_end, q_end, length.out = n_points)

  curve <- function(values, grid, series = "") {
    data.frame(x = grid, y = as.numeric(values), series = series)
  }

  if (flavour %in% c("lst", "grid", "reward")) {

    grid <- if (kind == "quantile") q_grid else seq(0, d$quantile(q_end), length.out = n_points)
    data <- curve(x(reticulate::np_array(grid)), grid)

    title <- switch(
      flavour,
      lst = d$`_titled`(switch(kind, pdf = "PDF", cdf = "CDF", quantile = "quantile function")),
      switch(kind, pdf = "PDF", cdf = "CDF", quantile = "Quantile function")
    )

    xlab <- switch(flavour, lst = "x", grid = "t", reward = "accumulated branch length")

    if (flavour == "grid") {
      ylab <- switch(kind, pdf = "f(t)", cdf = "F(t)", quantile = "quantile")
    }

    if (kind == "quantile") {
      xlab <- "q"
    }

    return(list(data = data, xlab = xlab, ylab = ylab, title = title, legend = NULL))
  }

  if (flavour == "sfs") {

    bins <- if (is.null(bins)) as.integer(d$`_get_indices`()) else as.integer(bins)
    dists <- lapply(bins, function(i) d$bin(i))

    grid <- if (kind == "quantile") {
      q_grid
    } else {
      seq(0, max(vapply(dists, function(b) b$quantile(q_end), numeric(1))), length.out = n_points)
    }

    evaluate <- function(b) {
      fun <- switch(kind, pdf = b$pdf, cdf = b$cdf, quantile = b$quantile)
      fun(reticulate::np_array(grid))
    }

    data <- do.call(rbind, Map(function(b, i) curve(evaluate(b), grid, i), dists, bins))

    title <- switch(kind, pdf = "SFS bin PDFs", cdf = "SFS bin CDFs", quantile = "SFS bin quantile functions")

    return(list(data = data, xlab = if (kind == "quantile") "q" else "accumulated branch length", ylab = ylab,
                title = title, legend = "bin"))
  }

  # empirical: a sample vector (scalar distribution) or a replicate-by-bin matrix (spectrum)
  shape <- as.integer(unlist(py_helpers()$samples_shape(d)))
  per_bin <- length(shape) == 2
  n_samples <- shape[1]

  if (per_bin && is.null(bins)) {
    bins <- seq_len(shape[2] - 2)
  }

  # row of each requested bin in the per-bin results, whose rows are the sample columns 0, ..., n
  select <- function(values, by_row = TRUE) {
    if (!per_bin) {
      return(list(as.numeric(values)))
    }
    lapply(bins, function(i) if (by_row) values[i + 1, ] else values[, i + 1])
  }

  if (kind == "quantile") {
    grid <- q_grid
    values <- select(x(reticulate::np_array(grid)), by_row = FALSE)
    xs <- rep(list(grid), length(values))
  } else {
    ends <- as.numeric(d$quantile(q_end))
    grid <- seq(0, if (per_bin) max(ends[bins + 1]) else ends, length.out = n_points)

    if (kind == "cdf") {
      values <- select(x(reticulate::np_array(grid)))
      xs <- rep(list(grid), length(values))
    } else {
      # the empirical density is a cell average, so the cells are coarsened with the sample size and the averages
      # are drawn at the cell centres
      cells <- seq(grid[1], grid[length(grid)], length.out = as.integer(min(max(sqrt(n_samples), 20), 100)))
      values <- select(x(reticulate::np_array(cells)))
      xs <- rep(list(cells + (cells[2] - cells[1]) / 2), length(values))
    }
  }

  series <- if (per_bin) bins else ""
  data <- do.call(rbind, Map(curve, values, xs, series))

  title <- if (per_bin) {
    switch(kind, pdf = "SFS bin PDFs", cdf = "SFS bin CDFs", quantile = "SFS bin quantile functions")
  } else {
    switch(kind, pdf = "PDF", cdf = "CDF", quantile = "Quantile function")
  }

  ylab <- switch(kind, pdf = "f(t)", cdf = "F(t)", quantile = "quantile")

  list(data = data, xlab = if (kind == "quantile") "q" else "t", ylab = ylab, title = title,
       legend = if (per_bin) "bin" else NULL)
}


# Curves of a univariate distribution function, labelled `label` and drawn into the plot `add` if given
plot_function <- function(x, kind, n_points, bins, title, label, add, linewidth, alpha) {

  curves <- function_curves(x, kind, n_points = n_points, bins = bins)
  data <- curves$data
  data$linewidth <- linewidth
  data$alpha <- alpha

  if (!is.null(label)) {
    data$series <- label
  }

  if (!is.null(add)) {
    previous <- add$data[c("x", "y", "series", "linewidth", "alpha")]
    previous$series <- as.character(previous$series)
    data <- rbind(previous, data)
  }

  line_plot(
    data,
    xlab = curves$xlab,
    ylab = curves$ylab,
    title = if (is.null(title)) curves$title else title,
    legend = curves$legend
  )
}


#' Plot a probability density function
#'
#' Draws the density of a distribution, such as `coal$tree_height$pdf`, a marginal or conditional density of a joint
#' distribution, or `coal$sfs$pdf` for all SFS bins at once.
#'
#' @param x The `pdf` of a distribution.
#' @param n_points Number of grid points, `NULL` for the default.
#' @param bins SFS bins to draw, `NULL` for all polymorphic bins.
#' @param title Plot title, `NULL` for the default.
#' @param label Legend label of the curve, `NULL` for none.
#' @param add A plot returned by a previous call to draw the curve into, `NULL` for a new plot.
#' @param linewidth Width of the curve.
#' @param alpha Opacity of the curve.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' coal <- pg$Coalescent(n = 10L)
#' plot(coal$tree_height$pdf)
#' plot(coal$total_branch_length$pdf, title = "Density")
#' }
#'
#' @method plot phasegen.distributions.base.DensityFunction
#' @export
plot.phasegen.distributions.base.DensityFunction <- function(x, n_points = NULL, bins = NULL, title = NULL,
                                                             label = NULL, add = NULL, linewidth = 0.5, alpha = 1,
                                                             ...) {
  plot_function(x, "pdf", n_points, bins, title, label, add, linewidth, alpha)
}


#' Plot a cumulative distribution function
#'
#' @param x The `cdf` of a distribution.
#' @param n_points Number of grid points, `NULL` for the default.
#' @param bins SFS bins to draw, `NULL` for all polymorphic bins.
#' @param title Plot title, `NULL` for the default.
#' @param label Legend label of the curve, `NULL` for none.
#' @param add A plot returned by a previous call to draw the curve into, `NULL` for a new plot.
#' @param linewidth Width of the curve.
#' @param alpha Opacity of the curve.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' plot(pg$Coalescent(n = 10L)$total_branch_length$cdf, title = "CDF")
#' }
#'
#' @method plot phasegen.distributions.base.CumulativeDistributionFunction
#' @export
plot.phasegen.distributions.base.CumulativeDistributionFunction <- function(x, n_points = NULL, bins = NULL,
                                                                             title = NULL, label = NULL,
                                                                             add = NULL, linewidth = 0.5,
                                                                             alpha = 1, ...) {
  plot_function(x, "cdf", n_points, bins, title, label, add, linewidth, alpha)
}


#' Plot a quantile function
#'
#' @param x The `quantile` function of a distribution.
#' @param n_points Number of grid points, `NULL` for the default.
#' @param bins SFS bins to draw, `NULL` for all polymorphic bins.
#' @param title Plot title, `NULL` for the default.
#' @param label Legend label of the curve, `NULL` for none.
#' @param add A plot returned by a previous call to draw the curve into, `NULL` for a new plot.
#' @param linewidth Width of the curve.
#' @param alpha Opacity of the curve.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' plot(pg$Coalescent(n = 10L)$total_branch_length$quantile, title = "Quantile")
#' }
#'
#' @method plot phasegen.distributions.base.QuantileFunction
#' @export
plot.phasegen.distributions.base.QuantileFunction <- function(x, n_points = NULL, bins = NULL, title = NULL,
                                                              label = NULL, add = NULL, linewidth = 0.5, alpha = 1,
                                                              ...) {
  plot_function(x, "quantile", n_points, bins, title, label, add, linewidth, alpha)
}


# ---- joint distribution functions -----------------------------------------------------------------------------------


# The joint density or CDF on phasegen's plotting grid: each axis runs from 0 to the marginal
# Settings.plot_endpoint_quantile quantile
joint_grid <- function(x, n_points, surface) {

  n_points <- if (is.null(n_points)) as.integer(x$`_default_n_points`(surface)) else as.integer(n_points)
  grid <- x$`_joint_grid`(n_points)
  xs <- as.numeric(grid[[1]])
  ys <- as.numeric(grid[[2]])
  z <- matrix(x(reticulate::np_array(xs), reticulate::np_array(ys)), length(xs), length(ys))

  list(x = xs, y = ys, z = z, is_cdf = inherits(x, "phasegen.distributions.base.CumulativeDistributionFunction"),
       title = x$`_joint_title`())
}


#' Plot a joint density or joint CDF as a heatmap
#'
#' @param x The `pdf` or `cdf` of a joint distribution.
#' @param n_points Number of grid points per axis, `NULL` for the default.
#' @param title Plot title, `NULL` for the default.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @seealso [persp.phasegen.distributions.base._JointFunction()] for the surface.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' joint <- pg$Coalescent(n = 8L)$sfs$joint_distribution(1L, 2L)
#' plot(joint$pdf, title = "Joint density")
#' }
#'
#' @method plot phasegen.distributions.base._JointFunction
#' @export
plot.phasegen.distributions.base._JointFunction <- function(x, n_points = NULL, title = NULL, ...) {

  g <- joint_grid(x, n_points, surface = FALSE)
  data <- expand.grid(x = g$x, y = g$y)
  data$z <- as.vector(g$z)

  ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$z)) +
    ggplot2::geom_raster() +
    ggplot2::scale_fill_viridis_c(limits = if (g$is_cdf) c(0, 1) else NULL) +
    ggplot2::coord_cartesian(expand = FALSE) +
    ggplot2::labs(
      x = quote(R[a]),
      y = quote(R[b]),
      fill = if (g$is_cdf) quote(F(R[a], R[b])) else quote(f(R[a], R[b])),
      title = if (is.null(title)) g$title else title
    ) +
    theme_phasegen()
}


#' Draw a joint density or joint CDF as a surface
#'
#' @param x The `pdf` or `cdf` of a joint distribution.
#' @param n_points Number of grid points per axis, `NULL` for the default.
#' @param title Plot title, `NULL` for the default.
#' @param theta,phi Viewing angles in degrees.
#' @param ... Further arguments passed to [graphics::persp()].
#'
#' @return The viewing transformation matrix, invisibly.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' joint <- pg$Coalescent(n = 8L)$sfs$joint_distribution(1L, 2L)
#' persp(joint$pdf, title = "Joint density")
#' persp(joint$cdf, title = "Joint CDF")
#' }
#'
#' @importFrom graphics persp
#' @method persp phasegen.distributions.base._JointFunction
#' @export
persp.phasegen.distributions.base._JointFunction <- function(x, n_points = NULL, title = NULL, theta = 30, phi = 30,
                                                             ...) {

  g <- joint_grid(x, n_points, surface = TRUE)

  draw_surface(
    g$x, g$y, g$z,
    xlab = "R_a", ylab = "R_b", zlab = if (g$is_cdf) "F(R_a, R_b)" else "f(R_a, R_b)",
    title = if (is.null(title)) g$title else title,
    zlim = if (g$is_cdf) c(0, 1) else NULL,
    theta = theta, phi = phi, ...
  )
}


# ---- spectra --------------------------------------------------------------------------------------------------------


# Grouped bar chart of spectra. `data` is a matrix with one column per spectrum (entries 0, ..., n).
bar_plot <- function(data, labels, title, log_scale, show_monomorphic, use_subplots = FALSE) {

  n <- nrow(data) - 1
  rows <- if (show_monomorphic) seq_len(n + 1) else seq_len(n - 1) + 1

  long <- data.frame(
    count = rep(rows - 1, times = ncol(data)),
    value = as.vector(data[rows, , drop = FALSE]),
    type = factor(rep(labels, each = length(rows)), levels = labels)
  )

  n_types <- length(labels)
  fill <- series_colours(n_types)
  show_legend <- n_types > 1 && !use_subplots

  # bars of the types side by side, together 0.9 wide and centred on each allele count
  width <- 0.9 / n_types
  long$xmin <- long$count - 0.45 + (as.integer(long$type) - 1) * width
  long$xmax <- long$xmin + width

  # on a logarithmic axis, which has no place for empty entries, bars rise from the bottom of the axis
  if (log_scale) {
    long$value[!(long$value > 0)] <- NA
    long$ymin <- min(long$value, na.rm = TRUE) / 2
  } else {
    long$ymin <- 0
  }

  p <- ggplot2::ggplot(long, ggplot2::aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin, ymax = .data$value,
                                          fill = .data$type, colour = .data$type)) +
    ggplot2::geom_rect(linewidth = 0.3, na.rm = TRUE) +
    ggplot2::scale_fill_manual(values = fill, guide = if (show_legend) "legend" else "none") +
    ggplot2::scale_colour_manual(values = darken(fill), guide = if (show_legend) "legend" else "none") +
    ggplot2::scale_x_continuous(
      breaks = function(limits) {
        breaks <- unique(round(pretty(limits, n = min(10, length(rows)))))
        breaks[breaks >= min(long$count) & breaks <= max(long$count)]
      },
      expand = ggplot2::expansion(add = 0.05)
    ) +
    ggplot2::labs(x = "allele count", y = NULL, title = title, fill = NULL, colour = NULL) +
    theme_phasegen()

  p <- p + if (log_scale) {
    ggplot2::scale_y_log10(expand = ggplot2::expansion(mult = c(0, 0.05)))
  } else {
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.05)))
  }

  # bars fill the panel below their tops, which leaves only the upper corners free
  if (show_legend) {
    p <- p + inside_legend(legend_corner(data.frame(x = long$count, y = long$value), corners = list(c(1, 1), c(0, 1))))
  }

  if (use_subplots) {
    p <- p + ggplot2::facet_wrap(ggplot2::vars(.data$type), scales = "free_y")
  }

  p
}


#' Plot a site-frequency spectrum
#'
#' @param x A spectrum, such as `coal$sfs$mean`.
#' @param title Plot title, `NULL` for none.
#' @param log_scale Logical, whether to use a logarithmic y-axis. Default is `FALSE`.
#' @param show_monomorphic Logical, whether to include the monomorphic entries 0 and n. Default is `FALSE`.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' plot(pg$Coalescent(n = 10L)$sfs$mean)
#' }
#'
#' @method plot sfsutils.spectrum.Spectrum
#' @export
plot.sfsutils.spectrum.Spectrum <- function(x, title = NULL, log_scale = FALSE, show_monomorphic = FALSE, ...) {
  bar_plot(matrix(as.numeric(x$data), ncol = 1), "", title, log_scale, show_monomorphic)
}


#' Plot grouped spectra
#'
#' @param x A `Spectra` object.
#' @param title Plot title, `NULL` for none.
#' @param log_scale Logical, whether to use a logarithmic y-axis. Default is `FALSE`.
#' @param use_subplots Logical, whether to draw each type in its own panel. Default is `FALSE`.
#' @param show_monomorphic Logical, whether to include the monomorphic entries 0 and n. Default is `FALSE`.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' coal <- pg$Coalescent(n = 10L)
#' plot(pg$Spectra(list(mean = coal$sfs$mean, sd = coal$sfs$std)))
#' }
#'
#' @method plot sfsutils.spectrum.Spectra
#' @export
plot.sfsutils.spectrum.Spectra <- function(x, title = NULL, log_scale = FALSE, use_subplots = FALSE,
                                           show_monomorphic = FALSE, ...) {

  data <- x$data

  bar_plot(as.matrix(data), colnames(data), title, log_scale, show_monomorphic, use_subplots)
}


# Symmetric logarithm of matplotlib's SymLogNorm (base 10, linscale 1): linear within [-threshold, threshold], where
# it spans as much of the scale as one decade, and logarithmic beyond
symlog <- function(x, threshold) {
  slope <- 1 / (1 - 1 / 10)
  ifelse(abs(x) <= threshold, x * slope, sign(x) * threshold * (slope + log10(abs(x) / threshold)))
}


# ColorBrewer's PuOr palette in the orientation of matplotlib's PuOr_r, purple for negative and orange for positive
puor_r <- rev(c("#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#f7f7f7", "#d8daeb", "#b2abd2", "#8073ac",
                "#542788", "#2d004b"))


# Interior entries of a 2-SFS matrix (the segregating bins), truncated to the folded bins for a folded spectrum
two_sfs_interior <- function(x) {

  data <- x$data
  n <- nrow(data) - 1

  if (n < 2) {
    return(NULL)
  }

  interior <- data[2:n, 2:n, drop = FALSE]

  if (x$is_folded()) {
    w <- as.integer(x$w)
    interior <- interior[seq_len(w - 1), seq_len(w - 1), drop = FALSE]
  }

  list(interior = interior, data = data)
}


#' Plot a 2-SFS as a heatmap
#'
#' Draws the segregating entries of a 2-SFS, such as `coal$sfs$corr` or `coal$sfs2$mean`, with entry (i, j) at x = j and
#' y = i. Covariances and correlations are drawn on a diverging colour scale centred at zero.
#'
#' @param x A `TwoSFS` object.
#' @param title Plot title, `NULL` for none.
#' @param max_abs Largest absolute value of the diverging colour scale, `NULL` for the largest absolute entry.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' plot(pg$Coalescent(n = 10L)$sfs$corr)
#' }
#'
#' @method plot sfsutils.spectrum.TwoSFS
#' @export
plot.sfsutils.spectrum.TwoSFS <- function(x, title = NULL, max_abs = NULL, ...) {

  parts <- two_sfs_interior(x)

  if (is.null(parts)) {
    stop("Nothing to plot: the 2-SFS has no segregating entries.", call. = FALSE)
  }

  interior <- parts$interior
  data <- parts$data
  m <- nrow(interior)

  long <- expand.grid(i = seq_len(m), j = seq_len(m))
  long$value <- as.vector(interior)

  is_counts <- sum(abs(c(data[c(1, nrow(data)), ], data[, c(1, ncol(data))])), na.rm = TRUE) > 0

  if (is_counts) {
    long$value[!(long$value > 0)] <- NA
    fill <- ggplot2::scale_fill_viridis_c(transform = "log10", na.value = NA)
  } else {
    if (is.null(max_abs)) {
      max_abs <- max(abs(interior), na.rm = TRUE)
      if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1
    }

    threshold <- max_abs / 10
    breaks <- c(-max_abs, -threshold, 0, threshold, max_abs)
    long$value <- symlog(pmin(pmax(long$value, -max_abs), max_abs), threshold)

    fill <- ggplot2::scale_fill_gradientn(
      colours = puor_r,
      limits = symlog(c(-max_abs, max_abs), threshold),
      breaks = symlog(breaks, threshold),
      labels = signif(breaks, 2),
      na.value = NA
    )
  }

  integer_breaks <- function(limits) {
    breaks <- unique(round(pretty(limits, n = min(m, 8))))
    breaks[breaks >= 1 & breaks <= m]
  }

  ggplot2::ggplot(long, ggplot2::aes(x = .data$j, y = .data$i, fill = .data$value)) +
    ggplot2::geom_raster(na.rm = TRUE) +
    fill +
    ggplot2::scale_x_continuous(breaks = integer_breaks, expand = c(0, 0)) +
    ggplot2::scale_y_continuous(breaks = integer_breaks, expand = c(0, 0)) +
    ggplot2::coord_fixed() +
    ggplot2::labs(x = NULL, y = NULL, fill = NULL, title = title) +
    theme_phasegen() +
    ggplot2::theme(panel.grid = ggplot2::element_blank(), panel.border = ggplot2::element_rect(colour = "grey50"))
}


#' Draw a 2-SFS as a surface
#'
#' Draws the segregating entries of a 2-SFS, with entry (i, j) at x = j and y = i.
#'
#' @param x A `TwoSFS` object.
#' @param title Plot title, `NULL` for none.
#' @param theta,phi Viewing angles in degrees.
#' @param ... Further arguments passed to [graphics::persp()].
#'
#' @return The viewing transformation matrix, invisibly.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' persp(pg$Coalescent(n = 10L)$sfs$cov, title = "Kingman")
#' }
#'
#' @method persp sfsutils.spectrum.TwoSFS
#' @export
persp.sfsutils.spectrum.TwoSFS <- function(x, title = NULL, theta = 30, phi = 30, ...) {

  parts <- two_sfs_interior(x)

  if (is.null(parts)) {
    stop("Nothing to plot: the 2-SFS has no segregating entries.", call. = FALSE)
  }

  m <- nrow(parts$interior)

  draw_surface(
    seq_len(m), seq_len(m), t(parts$interior),
    xlab = "allele count", ylab = "allele count", zlab = "", title = title, theta = theta, phi = phi, ...
  )
}


# A joint SFS marginalised onto two populations, with the monomorphic corners optionally masked
joint_sfs_data <- function(x, pops, mask_monomorphic) {

  pops <- as.integer(pops)

  if (length(pops) != 2) {
    stop("Exactly two populations must be specified.", call. = FALSE)
  }

  data <- x$marginalize(do.call(reticulate::tuple, as.list(pops)))$data

  if (length(dim(data)) != 2) {
    stop("Plotting requires a 2-dimensional (marginalized) joint SFS.", call. = FALSE)
  }

  if (mask_monomorphic) {
    data[1, 1] <- NA
    data[nrow(data), ncol(data)] <- NA
  }

  names <- as.character(x$`_names`())

  list(data = data, xlab = paste("allele count", names[pops[2] + 1]), ylab = paste("allele count", names[pops[1] + 1]))
}


#' Plot a joint SFS as a heatmap
#'
#' Draws the joint SFS of two populations, marginalised over any further populations.
#'
#' @param x A `JointSFS` object, such as `coal$jsfs$mean`.
#' @param pops Zero-based indices of the populations on the y-axis and the x-axis.
#' @param title Plot title, `NULL` for none.
#' @param log_scale Logical, whether to use a logarithmic colour scale. Default is `TRUE`.
#' @param mask_monomorphic Logical, whether to mask the monomorphic corners. Default is `TRUE`.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' plot(pg$Coalescent(n = list(pop_0 = 4L, pop_1 = 4L))$jsfs$mean)
#' }
#'
#' @method plot sfsutils.spectrum.JointSFS
#' @export
plot.sfsutils.spectrum.JointSFS <- function(x, pops = c(0, 1), title = NULL, log_scale = TRUE, mask_monomorphic = TRUE,
                                            ...) {

  j <- joint_sfs_data(x, pops, mask_monomorphic)
  long <- expand.grid(y = seq_len(nrow(j$data)) - 1, x = seq_len(ncol(j$data)) - 1)
  long$value <- as.vector(j$data)

  if (log_scale) {
    long$value[!(long$value > 0)] <- NA
  }

  ggplot2::ggplot(long, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$value)) +
    ggplot2::geom_raster(na.rm = TRUE) +
    ggplot2::scale_fill_viridis_c(transform = if (log_scale) "log10" else "identity", na.value = NA) +
    ggplot2::scale_x_continuous(breaks = function(l) unique(round(pretty(l))), expand = c(0, 0)) +
    ggplot2::scale_y_continuous(breaks = function(l) unique(round(pretty(l))), expand = c(0, 0)) +
    ggplot2::coord_fixed() +
    ggplot2::labs(x = j$xlab, y = j$ylab, fill = NULL, title = title) +
    theme_phasegen() +
    ggplot2::theme(panel.grid = ggplot2::element_blank(), panel.border = ggplot2::element_rect(colour = "grey50"))
}


#' Draw a joint SFS as a surface
#'
#' Draws the joint SFS of two populations, marginalised over any further populations.
#'
#' @param x A `JointSFS` object, such as `coal$jsfs$mean`.
#' @param pops Zero-based indices of the populations on the y-axis and the x-axis.
#' @param title Plot title, `NULL` for none.
#' @param mask_monomorphic Logical, whether to mask the monomorphic corners. Default is `TRUE`.
#' @param theta,phi Viewing angles in degrees.
#' @param ... Further arguments passed to [graphics::persp()].
#'
#' @return The viewing transformation matrix, invisibly.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' persp(pg$Coalescent(n = list(pop_0 = 4L, pop_1 = 4L))$jsfs$mean, title = "Mean joint SFS")
#' }
#'
#' @method persp sfsutils.spectrum.JointSFS
#' @export
persp.sfsutils.spectrum.JointSFS <- function(x, pops = c(0, 1), title = NULL, mask_monomorphic = TRUE, theta = 30,
                                             phi = 30, ...) {

  j <- joint_sfs_data(x, pops, mask_monomorphic)

  draw_surface(
    seq_len(ncol(j$data)) - 1, seq_len(nrow(j$data)) - 1, t(j$data),
    xlab = j$xlab, ylab = j$ylab, zlab = "branch length", title = title, theta = theta, phi = phi, ...
  )
}


# ---- demography -----------------------------------------------------------------------------------------------------


# Population sizes and migration rates of a demography at the times t, as a data frame with columns x, y and series
demography_rates <- function(d, t, which) {

  kinds <- switch(which, all = c("pop_sizes", "migration"), pop_sizes = "pop_sizes", migration = "migration")
  empty <- data.frame(x = numeric(0), y = numeric(0), series = character(0))

  do.call(rbind, c(list(empty), lapply(kinds, function(kind) {
    result <- py_helpers()$rates(d, reticulate::np_array(t), kind)
    names <- as.character(unlist(result[[1]]))

    if (length(names) == 0) {
      return(NULL)
    }

    values <- matrix(result[[2]], nrow = length(t))

    data.frame(x = rep(t, times = length(names)), y = as.vector(values), series = rep(names, each = length(t)))
  })))
}


# default title and y-axis label of each demography panel
demography_labels <- function(which) {
  switch(
    which,
    all = list(title = "Demography", ylab = quote(N[e] * "," ~ m[ij])),
    pop_sizes = list(title = "Population size trajectory", ylab = quote(N[e])),
    migration = list(title = "Migration rate trajectory", ylab = quote(m[ij]))
  )
}


#' Plot a demography
#'
#' @param x A `Demography` object, such as `coal$demography`.
#' @param which What to draw: `"all"` for population sizes and migration rates, `"pop_sizes"` or `"migration"`.
#' @param t Times at which to evaluate the trajectories, `NULL` for 1000 points from 0 to 10.
#' @param title Plot title, `NULL` for the default.
#' @param ylab Label of the y-axis, `NULL` for the default.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' d <- pg$Demography(pop_sizes = list(pop_0 = 1, pop_1 = 2))
#' plot(d)
#' plot(d, which = "migration")
#' }
#'
#' @method plot phasegen.demography.Demography
#' @export
plot.phasegen.demography.Demography <- function(x, which = c("all", "pop_sizes", "migration"), t = NULL, title = NULL,
                                                ylab = NULL, ...) {

  which <- match.arg(which)
  t <- if (is.null(t)) seq(0, 10, length.out = 1000) else as.numeric(t)
  labels <- demography_labels(which)
  data <- demography_rates(x, t, which)

  line_plot(
    data,
    xlab = "t",
    ylab = if (is.null(ylab)) labels$ylab else ylab,
    title = if (is.null(title)) labels$title else title,
    step = TRUE
  )
}


# ---- moment accumulation --------------------------------------------------------------------------------------------


#' Plot the accumulation of a moment over time
#'
#' Draws a moment of a phase-type distribution with the absorption time truncated at each end time, one curve per bin
#' for an SFS distribution.
#'
#' @param x A phase-type distribution, such as `coal$tree_height` or `coal$sfs`.
#' @param k The order of the moment. Default is `1`.
#' @param end_times Times at which to evaluate the moment, `NULL` for the default.
#' @param rewards A list of `k` rewards, `NULL` for the reward of the distribution.
#' @param center Logical, whether to center the moment around the mean. Default is `TRUE`.
#' @param permute Logical, whether to average cross-moments over all permutations of the rewards. Default is `TRUE`.
#' @param title Plot title, `NULL` for the default.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' coal <- pg$Coalescent(n = 10L)
#' plot_accumulation(coal$sfs, k = 1L)
#' plot_accumulation(coal$tree_height, k = 2L)
#' }
#'
#' @export
plot_accumulation <- function(x, ...) {
  UseMethod("plot_accumulation")
}


# moment of the distribution at each end time, with the default grid and title of the Python method
accumulation <- function(x, k, end_times, rewards, center, permute, title, prefix) {

  k <- as.integer(k)
  settings <- plot_settings()

  if (is.null(end_times)) {
    end_times <- seq(0, x$tree_height$quantile(settings$q_end), length.out = settings$n_grid)
  }

  if (is.null(title)) {
    reward_list <- if (is.null(rewards)) rep(list(x$reward), k) else rewards
    names <- vapply(reward_list, function(r) sub("Reward", "", py_helpers()$type_name(r)), character(1))
    title <- paste0(prefix, " (", paste(names, collapse = ", "), ")")
  }

  values <- x$accumulate(k, reticulate::np_array(as.numeric(end_times)), rewards, center, permute)

  list(t = as.numeric(end_times), values = values, title = title)
}


#' @rdname plot_accumulation
#' @method plot_accumulation phasegen.distributions.phase_type.PhaseTypeDistribution
#' @export
plot_accumulation.phasegen.distributions.phase_type.PhaseTypeDistribution <- function(x, k = 1L, end_times = NULL,
                                                                                      rewards = NULL, center = TRUE,
                                                                                      permute = TRUE, title = NULL,
                                                                                      ...) {

  acc <- accumulation(x, k, end_times, rewards, center, permute, title, "Moment accumulation")

  line_plot(data.frame(x = acc$t, y = as.numeric(acc$values), series = ""), xlab = "t", ylab = "moment",
            title = acc$title)
}


#' @rdname plot_accumulation
#' @method plot_accumulation phasegen.distributions.spectra.SFSDistribution
#' @export
plot_accumulation.phasegen.distributions.spectra.SFSDistribution <- function(x, k = 1L, end_times = NULL,
                                                                             rewards = NULL, center = TRUE,
                                                                             permute = TRUE, title = NULL, ...) {

  acc <- accumulation(x, k, end_times, rewards, center, permute, title, "SFS Moment accumulation")
  bins <- as.integer(x$`_get_indices`())

  data <- do.call(rbind, lapply(bins, function(i) data.frame(x = acc$t, y = acc$values[i + 1, ], series = i)))

  line_plot(data, xlab = "t", ylab = "moment", title = acc$title, legend = "bin")
}


# ---- inference ------------------------------------------------------------------------------------------------------


#' Plot the results of a demographic inference
#'
#' Draws the inferred demography together with its bootstrapped trajectories, or the distributions of the bootstrapped
#' parameters.
#'
#' @param x An `Inference` object after `x$run()`.
#' @param which What to draw: `"demography"` for population sizes and migration rates, `"pop_sizes"`, `"migration"`,
#'        or `"bootstraps"` for the bootstrapped parameters.
#' @param t Times at which to evaluate the trajectories, `NULL` for 100 points up to the 0.99 quantile of the inferred
#'        tree height.
#' @param include_bootstraps Logical, whether to draw the bootstrapped trajectories. Default is `TRUE`.
#' @param kind For `which = "bootstraps"`, `"hist"` for histograms or `"kde"` for kernel density estimates.
#' @param bins Number of histogram bins. Default is `20`.
#' @param title Plot title, `NULL` for the default.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' inf$run()
#' plot(inf, which = "pop_sizes")
#' plot(inf, which = "bootstraps")
#' }
#'
#' @method plot phasegen.inference.Inference
#' @export
plot.phasegen.inference.Inference <- function(x, which = c("demography", "pop_sizes", "migration", "bootstraps"),
                                              t = NULL, include_bootstraps = TRUE, kind = c("hist", "kde"),
                                              bins = 20L, title = NULL, ...) {

  which <- match.arg(which)

  if (which == "bootstraps") {
    return(plot_bootstraps(x, match.arg(kind), bins, title))
  }

  if (is.null(x$dist_inferred)) {
    stop("The main optimization must be run first (call inf$run()).", call. = FALSE)
  }

  if (is.null(t)) {
    t <- seq(0, x$dist_inferred$tree_height$quantile(0.99), length.out = 100)
  }

  t <- as.numeric(t)
  kind_rates <- if (which == "demography") "all" else which
  labels <- demography_labels(kind_rates)

  data <- demography_rates(x$dist_inferred$demography, t, kind_rates)
  data$series <- factor(data$series, levels = unique(data$series))

  replicates <- if (include_bootstraps) py_helpers()$bootstrap_demographies(x) else list()
  under <- NULL

  if (length(replicates) > 0) {

    boots <- do.call(rbind, Map(function(d, i) {
      rates <- demography_rates(d, t, kind_rates)
      rates$replicate <- rep(i, nrow(rates))
      rates
    }, replicates, seq_along(replicates)))

    boots$series <- factor(boots$series, levels = levels(data$series))

    under <- ggplot2::geom_step(
      data = boots,
      ggplot2::aes(group = interaction(.data$series, .data$replicate)),
      direction = "hv",
      alpha = 0.3
    )
  }

  line_plot(data, xlab = "t", ylab = labels$ylab, title = if (is.null(title)) labels$title else title, step = TRUE,
            under = under)
}


# histograms or density estimates of the bootstrapped parameters, one panel per parameter
plot_bootstraps <- function(x, kind, bins, title) {

  result <- py_helpers()$bootstraps(x)
  names <- as.character(unlist(result[[1]]))

  if (is.null(result[[2]])) {
    stop("No bootstraps available.", call. = FALSE)
  }

  values <- matrix(result[[2]], ncol = length(names))

  data <- data.frame(
    parameter = factor(rep(names, each = nrow(values)), levels = names),
    value = as.vector(values)
  )

  p <- ggplot2::ggplot(data, ggplot2::aes(x = .data$value, fill = .data$parameter, colour = .data$parameter))

  p <- p + if (kind == "hist") {
    ggplot2::geom_histogram(bins = as.integer(bins), alpha = 0.8, linewidth = 0.2)
  } else {
    ggplot2::geom_density(alpha = 0.4)
  }

  p +
    ggplot2::facet_wrap(ggplot2::vars(.data$parameter), ncol = 1, scales = "free") +
    ggplot2::scale_fill_manual(values = series_colours(length(names)), guide = "none") +
    ggplot2::scale_colour_manual(values = darken(series_colours(length(names))), guide = "none") +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.05))) +
    ggplot2::labs(x = NULL, y = if (kind == "hist") "count" else "density",
                  title = if (is.null(title)) "Marginal distributions" else title) +
    theme_phasegen()
}
