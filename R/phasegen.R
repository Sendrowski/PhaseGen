#' Check if the `phasegen` Python module is installed
#'
#' This function uses the reticulate package to verify if the `phasegen` Python
#' module is currently installed.
#'
#' @return Logical `TRUE` if the `phasegen` Python module is installed, otherwise `FALSE`.
#'
#' @examples
#' \dontrun{
#' phasegen_is_installed()  # Returns TRUE or FALSE based on the installation status of phasegen
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
#' Loading the package declares `phasegen`. This function declares a pinned version. The requirement is resolved when
#' Python is first initialised, at which point reticulate provisions an environment satisfying it.
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
import io

class RStream(io.TextIOBase):
    encoding = 'utf-8'

    def __init__(self, write, lines=False):
        super().__init__()
        self._write = write
        self._lines = lines
        self._buffer = ''

    def writable(self):
        return True

    def write(self, text):
        if not self._lines:
            self._write(text)
            return len(text)

        *complete, partial = (self._buffer + text).split('\\n')
        for line in complete:
            self._write(line.split('\\r')[-1])

        # a carriage return starts the line over, as a progress bar redraws itself
        self._buffer = partial.split('\\r')[-1]

        return len(text)
", local = TRUE, convert = FALSE)

  sys <- reticulate::import("sys", convert = FALSE)
  sys$stdout <- streams$RStream(function(text) cat(reticulate::py_to_r(text)))
  sys$stderr <- streams$RStream(
    function(line) message(reticulate::py_to_r(line), appendLF = FALSE),
    lines = TRUE
  )

  invisible(NULL)
}


# ---- marginal deme distributions ------------------------------------------------------------------------------------


# The marginal distribution of the deme `name`, NULL if no deme has that name
deme <- function(x, name) {

  if (is.character(name) && length(name) == 1 && name %in% reticulate::import_builtins()$list(x)) {
    reticulate::py_get_item(x, name)
  }
}


#' Access the marginal distribution of a deme
#'
#' Looks up the distribution of a deme by name, such as `coal$sfs$demes$pop_0`, and any other attribute otherwise.
#'
#' @param x The marginal deme distributions, such as `coal$sfs$demes`.
#' @param name Name of a deme or an attribute.
#'
#' @return The marginal distribution of the deme, or the attribute.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' coal <- pg$Coalescent(
#'   n = list(pop_0 = 3L, pop_1 = 2L),
#'   demography = pg$Demography(
#'     pop_sizes = list(pop_0 = 1, pop_1 = 1),
#'     events = c(pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1))
#'   )
#' )
#' coal$sfs$demes$pop_0$mean
#' coal$sfs$demes[["pop_1"]]$mean
#' }
#'
#' @method $ phasegen.distributions.base.MarginalDemeDistributions
#' @export
`$.phasegen.distributions.base.MarginalDemeDistributions` <- function(x, name) {
  d <- deme(x, name)
  if (is.null(d)) NextMethod() else d
}


#' @rdname cash-.phasegen.distributions.base.MarginalDemeDistributions
#' @method [[ phasegen.distributions.base.MarginalDemeDistributions
#' @export
`[[.phasegen.distributions.base.MarginalDemeDistributions` <- function(x, name) {
  d <- deme(x, name)
  if (is.null(d)) NextMethod() else d
}


# R-native plotting of phasegen objects: ggplot2 for 2D figures, graphics::persp() for surfaces. The methods dispatch
# on the class vectors reticulate assigns to Python objects, so plot(coal$sfs$mean) and persp(joint$pdf) work directly.
# Grids, values and labels come from the plot data methods of the Python objects.


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


# stops on arguments that no parameter of the calling method matched
check_unused <- function(...) {

  if (...length() > 0) {
    names <- names(list(...))
    names <- if (is.null(names)) rep("<unnamed>", ...length()) else replace(names, names == "", "<unnamed>")
    stop("Unused argument(s): ", paste(names, collapse = ", "), call. = FALSE)
  }
}


# numpy array of an R vector, NULL for NULL
np_or_null <- function(x) {
  if (is.null(x)) NULL else reticulate::np_array(as.numeric(x))
}


# Axis label of a Python plot as a plotmath expression, so that subscripts such as '$N_e$' or 'f(R_a, R_b)' render as
# such; labels that do not parse stay text
math_label <- function(label) {

  if (is.null(label) || !grepl("[$_]", label)) {
    return(label)
  }

  text <- gsub("[$]", "", label)
  math <- gsub("_(\\w)", "[\\1]", gsub("_\\{([^}]*)\\}", "[\\1]", text))
  math <- gsub(",\\s*", " * ',' ~ ", math)

  tryCatch(str2lang(math), error = function(e) text)
}


# Axis label of a Python plot as plain text, for persp()
text_label <- function(label) {
  gsub("[${}]", "", label)
}


# breaks at integers within the limits
integer_breaks <- function(n = 8) {
  function(limits) {
    breaks <- unique(round(pretty(limits, n = n)))
    breaks[breaks >= limits[1] & breaks <= limits[2]]
  }
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


# The curves of a Python _CurveData as a data frame with columns x, y, series (the curve labels) and curve (the index of
# each curve)
curve_frame <- function(curves) {

  x <- as.numeric(curves$x)
  labels <- as.character(unlist(curves$labels))
  y <- matrix(as.numeric(curves$y), ncol = length(x))

  data.frame(
    x = rep(x, each = length(labels)),
    y = as.vector(y),
    series = rep(labels, times = length(x)),
    curve = rep(seq_along(labels), times = length(x))
  )
}


# Line plot of one or more curves. `data` has columns x, y, series and curve, where series is a factor or is converted
# to one in order of appearance. Each curve is drawn as its own line, coloured by series, with a legend for more than
# one series. Layers in `under` are drawn beneath the curves.
line_plot <- function(data, xlab, ylab, title, legend = NULL, step = FALSE, under = NULL) {

  if (!is.factor(data$series)) {
    data$series <- factor(data$series, levels = unique(data$series))
  }

  n_series <- nlevels(data$series)

  if (is.null(data$linewidth)) {
    data$linewidth <- rep(0.5, nrow(data))
  }

  if (is.null(data$alpha)) {
    data$alpha <- rep(1, nrow(data))
  }

  style <- ggplot2::aes(linewidth = .data$linewidth, alpha = .data$alpha)
  geom <- if (step) ggplot2::geom_step(style, direction = "hv") else ggplot2::geom_line(style)

  p <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y, colour = .data$series, group = .data$curve)) +
    under +
    geom +
    ggplot2::scale_colour_manual(values = series_colours(n_series), drop = FALSE,
                                 guide = if (n_series > 1) "legend" else "none") +
    ggplot2::scale_linewidth_identity() +
    ggplot2::scale_alpha_identity() +
    ggplot2::scale_x_continuous(expand = c(0, 0)) +
    ggplot2::labs(x = math_label(xlab), y = math_label(ylab), title = title, colour = legend) +
    theme_phasegen()

  if (n_series > 1) {
    p <- p + inside_legend(legend_corner(data))
  }

  p
}


# Heatmap of z over the grid x by y (z has dimension length(x) by length(y)) with the given fill scale. Integer axes
# have integer breaks, a fixed aspect ratio and a panel border.
heatmap_plot <- function(x, y, z, fill, xlab, ylab, title, filllab = NULL, integer = FALSE) {

  data <- expand.grid(x = x, y = y)
  data$z <- as.vector(z)

  p <- ggplot2::ggplot(data, ggplot2::aes(x = .data$x, y = .data$y, fill = .data$z)) +
    ggplot2::geom_raster(na.rm = TRUE) +
    fill +
    ggplot2::labs(x = math_label(xlab), y = math_label(ylab), fill = math_label(filllab), title = title) +
    theme_phasegen()

  if (!integer) {
    return(p + ggplot2::coord_cartesian(expand = FALSE))
  }

  p +
    ggplot2::scale_x_continuous(breaks = integer_breaks(), expand = c(0, 0)) +
    ggplot2::scale_y_continuous(breaks = integer_breaks(), expand = c(0, 0)) +
    ggplot2::coord_fixed() +
    ggplot2::theme(panel.border = ggplot2::element_rect(colour = "grey50"))
}


# Surface of z over the grid x by y (z has dimension length(x) by length(y)) drawn with persp(), each facet coloured
# by its mean height on the viridis palette. `user` holds further arguments of persp(), which take precedence over
# `defaults`.
draw_surface <- function(x, y, z, defaults, user = list(), n_colours = 100) {

  nx <- length(x)
  ny <- length(y)

  if (nx < 2 || ny < 2) {
    stop("Drawing a surface requires a grid of at least 2 x 2 values.", call. = FALSE)
  }

  args <- utils::modifyList(c(defaults, list(ticktype = "detailed", cex.axis = 0.6, cex.lab = 0.8)), user)

  facets <- (z[-1, -1] + z[-1, -ny] + z[-nx, -1] + z[-nx, -ny]) / 4

  # the palette spans the facet heights, or a given zlim
  colour_range <- if (is.null(args$zlim)) range(facets, finite = TRUE) else args$zlim

  if (is.null(args$zlim)) {
    args$zlim <- range(z[is.finite(z)])
  }

  if (diff(args$zlim) == 0) {
    args$zlim <- args$zlim + c(-0.5, 0.5)
  }

  if (diff(colour_range) == 0) {
    colour_range <- colour_range + c(-0.5, 0.5)
  }

  if (is.null(args$col)) {
    facets <- pmin(pmax(facets, colour_range[1]), colour_range[2])
    breaks <- seq(colour_range[1], colour_range[2], length.out = n_colours + 1)
    args$col <- grDevices::hcl.colors(n_colours, "viridis")[cut(facets, breaks, include.lowest = TRUE, labels = FALSE)]
  }

  # facets are outlined on coarse grids only, as outlines on a fine grid cover the surface
  if (is.null(args$border)) {
    args$border <- if (max(nx, ny) > 25) NA else "grey20"
  }

  # persp() places the axis titles close to the tick labels: small labels and a leading line break, which moves each
  # title one line outwards, keep them apart
  for (lab in c("xlab", "ylab", "zlab")) {
    if (is.character(args[[lab]]) && nzchar(args[[lab]])) {
      args[[lab]] <- paste0("\n", args[[lab]])
    }
  }

  # the default margins leave the projected box small within the figure
  op <- graphics::par(mar = c(1.8, 0.8, 1.8, 0.8))
  on.exit(graphics::par(op))

  invisible(do.call(graphics::persp, c(list(x = x, y = y, z = z), args)))
}


# ---- mutational configurations --------------------------------------------------------------------------------------


#' Convert a mutational configuration to R
#'
#' @param x A `MutationConfig`, as yielded by `coal$sfs$get_mutation_configs()`.
#'
#' @return The counts in bin order, an integer vector.
#'
#' @exportS3Method reticulate::py_to_r
py_to_r.phasegen.distributions.mutation_configs.MutationConfig <- function(x) {
  as.integer(reticulate::import_builtins()$list(x))
}


# ---- univariate distribution functions ------------------------------------------------------------------------------


# Curves of a univariate distribution function over the grid `grid` (named `t`, or `q` for a quantile function), labelled
# `label` and drawn into the plot `add` if given
plot_function <- function(x, grid, n_points, bins, configs, title, label, add, linewidth, alpha) {

  args <- list()
  args[[if (inherits(x, "phasegen.distributions.base.QuantileFunction")) "q" else "t"]] <-
    if (!is.null(grid)) reticulate::np_array(as.numeric(grid))
  args$n_points <- if (!is.null(n_points)) as.integer(n_points)
  args$bins <- if (!is.null(bins)) as.integer(bins)
  args$configs <- if (!is.null(configs)) lapply(configs, as.integer)

  curves <- do.call(x$`_plot_data`, args)
  data <- curve_frame(curves)
  legend <- curves$legend_title

  if (!is.null(label)) {
    data$series <- if (nrow(data) > 0 && max(data$curve) > 1) {
      paste0(label, " (", paste(legend, data$series), ")")
    } else {
      rep(label, nrow(data))
    }
    legend <- NULL
  }

  data$linewidth <- linewidth
  data$alpha <- alpha

  if (!is.null(add)) {
    columns <- names(data)

    if (!inherits(add, "ggplot") || !all(columns %in% names(add$data))) {
      stop("'add' must be a plot returned by plot() of a distribution function.", call. = FALSE)
    }

    previous <- add$data[columns]
    previous$series <- as.character(previous$series)
    data$curve <- data$curve + max(previous$curve, 0)
    data <- rbind(previous, data)
  }

  line_plot(
    data,
    xlab = curves$xlabel,
    ylab = curves$ylabel,
    title = if (is.null(title)) curves$title else title,
    legend = legend
  )
}


#' Plot a probability density function
#'
#' Draws the density of a distribution, such as `coal$tree_height$pdf`, a marginal or conditional density of a joint
#' distribution, or `coal$sfs$pdf` for all SFS bins at once.
#'
#' @param x The `pdf` of a distribution.
#' @param t Points to evaluate at, `NULL` for the default grid.
#' @param n_points Number of grid points, `NULL` for the default.
#' @param bins SFS bins to draw, `NULL` for all polymorphic bins.
#' @param configs Joint SFS bins to draw, a list of integer vectors, `NULL` for all.
#' @param title Plot title, `NULL` for the default.
#' @param label Legend label of the curves, `NULL` for none.
#' @param add A plot returned by a previous call to draw the curves into, `NULL` for a new plot.
#' @param linewidth Width of the curves.
#' @param alpha Opacity of the curves.
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
plot.phasegen.distributions.base.DensityFunction <- function(x, t = NULL, n_points = NULL, bins = NULL, configs = NULL,
                                                             title = NULL, label = NULL, add = NULL, linewidth = 0.5,
                                                             alpha = 1, ...) {
  check_unused(...)
  plot_function(x, t, n_points, bins, configs, title, label, add, linewidth, alpha)
}


#' Plot a cumulative distribution function
#'
#' @param x The `cdf` of a distribution.
#' @param t Points to evaluate at, `NULL` for the default grid.
#' @param n_points Number of grid points, `NULL` for the default.
#' @param bins SFS bins to draw, `NULL` for all polymorphic bins.
#' @param configs Joint SFS bins to draw, a list of integer vectors, `NULL` for all.
#' @param title Plot title, `NULL` for the default.
#' @param label Legend label of the curves, `NULL` for none.
#' @param add A plot returned by a previous call to draw the curves into, `NULL` for a new plot.
#' @param linewidth Width of the curves.
#' @param alpha Opacity of the curves.
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
plot.phasegen.distributions.base.CumulativeDistributionFunction <- function(x, t = NULL, n_points = NULL, bins = NULL,
                                                                             configs = NULL, title = NULL, label = NULL,
                                                                             add = NULL, linewidth = 0.5,
                                                                             alpha = 1, ...) {
  check_unused(...)
  plot_function(x, t, n_points, bins, configs, title, label, add, linewidth, alpha)
}


#' Plot a quantile function
#'
#' @param x The `quantile` function of a distribution.
#' @param q Probabilities to evaluate at, `NULL` for the default grid.
#' @param n_points Number of grid points, `NULL` for the default.
#' @param bins SFS bins to draw, `NULL` for all polymorphic bins.
#' @param configs Joint SFS bins to draw, a list of integer vectors, `NULL` for all.
#' @param title Plot title, `NULL` for the default.
#' @param label Legend label of the curves, `NULL` for none.
#' @param add A plot returned by a previous call to draw the curves into, `NULL` for a new plot.
#' @param linewidth Width of the curves.
#' @param alpha Opacity of the curves.
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
plot.phasegen.distributions.base.QuantileFunction <- function(x, q = NULL, n_points = NULL, bins = NULL, configs = NULL,
                                                              title = NULL, label = NULL, add = NULL, linewidth = 0.5,
                                                              alpha = 1, ...) {
  check_unused(...)
  plot_function(x, q, n_points, bins, configs, title, label, add, linewidth, alpha)
}


# ---- joint distribution functions -----------------------------------------------------------------------------------


# The grid and values of a joint density or CDF
joint_data <- function(x, n_points, surface) {

  data <- x$`_plot_data`(n_points = if (!is.null(n_points)) as.integer(n_points), surface = surface)
  xs <- as.numeric(data$x)
  ys <- as.numeric(data$y)

  list(x = xs, y = ys, z = matrix(as.numeric(data$z), length(xs), length(ys)), data = data,
       zlim = if (!is.null(data$vmin) && !is.null(data$vmax)) c(data$vmin, data$vmax))
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

  check_unused(...)
  g <- joint_data(x, n_points, surface = FALSE)

  heatmap_plot(
    g$x, g$y, g$z,
    fill = ggplot2::scale_fill_viridis_c(limits = g$zlim),
    xlab = g$data$xlabel, ylab = g$data$ylabel, filllab = g$data$zlabel,
    title = if (is.null(title)) g$data$title else title
  )
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

  g <- joint_data(x, n_points, surface = TRUE)

  draw_surface(
    g$x, g$y, g$z,
    defaults = list(xlab = text_label(g$data$xlabel), ylab = text_label(g$data$ylabel),
                    zlab = text_label(g$data$zlabel), main = if (is.null(title)) g$data$title else title,
                    zlim = g$zlim, theta = theta, phi = phi),
    user = list(...)
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
    ggplot2::scale_x_continuous(breaks = integer_breaks(min(10, length(rows))),
                                expand = ggplot2::expansion(add = 0.05)) +
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
  check_unused(...)
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
  check_unused(...)
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


# Segregating entries of a 2-SFS matrix, truncated to the folded bins for a folded spectrum
two_sfs_interior <- function(data, folded, w) {

  n <- nrow(data) - 1
  bins <- seq_len(max(n - 1, 0)) + 1

  if (folded) {
    bins <- bins[seq_len(max(w - 1, 0))]
  }

  if (length(bins) == 0) {
    stop("Nothing to plot: the 2-SFS has no segregating entries.", call. = FALSE)
  }

  data[bins, bins, drop = FALSE]
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

  check_unused(...)
  data <- x$data
  interior <- two_sfs_interior(data, x$is_folded(), x$w)
  m <- nrow(interior)

  is_counts <- sum(abs(c(data[c(1, nrow(data)), ], data[, c(1, ncol(data))])), na.rm = TRUE) > 0

  if (is_counts) {
    values <- replace(interior, !(interior > 0), NA)
    fill <- ggplot2::scale_fill_viridis_c(transform = "log10", na.value = NA)
  } else {
    if (is.null(max_abs)) {
      max_abs <- max(abs(interior), na.rm = TRUE)
      if (!is.finite(max_abs) || max_abs == 0) max_abs <- 1
    }

    threshold <- max_abs / 10
    breaks <- c(-max_abs, -threshold, 0, threshold, max_abs)
    values <- symlog(pmin(pmax(interior, -max_abs), max_abs), threshold)

    fill <- ggplot2::scale_fill_gradientn(
      colours = puor_r,
      limits = symlog(c(-max_abs, max_abs), threshold),
      breaks = symlog(breaks, threshold),
      labels = signif(breaks, 2),
      na.value = NA
    )
  }

  heatmap_plot(seq_len(m), seq_len(m), t(values), fill, xlab = NULL, ylab = NULL, title = title, integer = TRUE)
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

  interior <- two_sfs_interior(x$data, x$is_folded(), x$w)
  m <- nrow(interior)

  draw_surface(
    seq_len(m), seq_len(m), t(interior),
    defaults = list(xlab = "allele count", ylab = "allele count", zlab = "", main = title, theta = theta, phi = phi),
    user = list(...)
  )
}


# A joint SFS marginalised onto two populations, with the monomorphic corners optionally masked
joint_sfs_data <- function(x, pops, mask_monomorphic) {

  if (length(pops) != 2) {
    stop("Exactly two populations must be specified.", call. = FALSE)
  }

  marginal <- x$marginalize(do.call(reticulate::tuple, as.list(as.integer(pops))))
  data <- marginal$data
  names <- as.character(unlist(marginal$pop_names))

  if (mask_monomorphic) {
    data[1, 1] <- NA
    data[nrow(data), ncol(data)] <- NA
  }

  list(data = data, xlab = paste("allele count", names[2]), ylab = paste("allele count", names[1]))
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
#' coal <- pg$Coalescent(
#'   n = list(pop_0 = 3L, pop_1 = 3L),
#'   demography = pg$Demography(
#'     pop_sizes = list(pop_0 = 1, pop_1 = 1),
#'     events = c(pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1))
#'   )
#' )
#' plot(coal$jsfs$mean)
#' }
#'
#' @method plot sfsutils.spectrum.JointSFS
#' @export
plot.sfsutils.spectrum.JointSFS <- function(x, pops = c(0, 1), title = NULL, log_scale = TRUE, mask_monomorphic = TRUE,
                                            ...) {

  check_unused(...)
  j <- joint_sfs_data(x, pops, mask_monomorphic)
  values <- if (log_scale) replace(j$data, !(j$data > 0), NA) else j$data

  heatmap_plot(
    seq_len(ncol(values)) - 1, seq_len(nrow(values)) - 1, t(values),
    fill = ggplot2::scale_fill_viridis_c(transform = if (log_scale) "log10" else "identity", na.value = NA),
    xlab = j$xlab, ylab = j$ylab, title = title, integer = TRUE
  )
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
#' coal <- pg$Coalescent(
#'   n = list(pop_0 = 3L, pop_1 = 3L),
#'   demography = pg$Demography(
#'     pop_sizes = list(pop_0 = 1, pop_1 = 1),
#'     events = c(pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1))
#'   )
#' )
#' persp(coal$jsfs$mean, title = "Mean joint SFS")
#' }
#'
#' @method persp sfsutils.spectrum.JointSFS
#' @export
persp.sfsutils.spectrum.JointSFS <- function(x, pops = c(0, 1), title = NULL, mask_monomorphic = TRUE, theta = 30,
                                             phi = 30, ...) {

  j <- joint_sfs_data(x, pops, mask_monomorphic)

  draw_surface(
    seq_len(ncol(j$data)) - 1, seq_len(nrow(j$data)) - 1, t(j$data),
    defaults = list(xlab = j$xlab, ylab = j$ylab, zlab = "branch length", main = title, theta = theta, phi = phi),
    user = list(...)
  )
}


# ---- demography -----------------------------------------------------------------------------------------------------


#' Plot a demography
#'
#' @param x A `Demography` object, such as `coal$demography`.
#' @param which What to draw: `"all"` for population sizes and migration rates, `"pop_sizes"` or `"migration"`.
#' @param t Times at which to evaluate the trajectories, `NULL` for the default.
#' @param title Plot title, `NULL` for the default.
#' @param ylab Label of the y-axis, `NULL` for the default.
#' @param ... Unused.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' pg <- load_phasegen()
#' d <- pg$Demography(
#'   pop_sizes = list(pop_0 = 1, pop_1 = 2),
#'   events = c(pg$MigrationRateChange(source = "pop_0", dest = "pop_1", time = 0, rate = 1))
#' )
#' plot(d)
#' plot(d, which = "migration")
#' }
#'
#' @method plot phasegen.demography.Demography
#' @export
plot.phasegen.demography.Demography <- function(x, which = c("all", "pop_sizes", "migration"), t = NULL, title = NULL,
                                                ylab = NULL, ...) {

  check_unused(...)
  curves <- x$`_plot_data`(np_or_null(t), match.arg(which))

  line_plot(
    curve_frame(curves),
    xlab = curves$xlabel,
    ylab = if (is.null(ylab)) curves$ylabel else ylab,
    title = if (is.null(title)) curves$title else title,
    step = TRUE
  )
}


# ---- moment accumulation --------------------------------------------------------------------------------------------


#' Plot the accumulation of a moment over time
#'
#' Draws a moment of a phase-type distribution with the absorption time truncated at each end time, one curve per bin
#' for a spectrum distribution.
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


#' @rdname plot_accumulation
#' @method plot_accumulation phasegen.distributions.phase_type.PhaseTypeDistribution
#' @export
plot_accumulation.phasegen.distributions.phase_type.PhaseTypeDistribution <- function(x, k = 1L, end_times = NULL,
                                                                                      rewards = NULL, center = TRUE,
                                                                                      permute = TRUE, title = NULL,
                                                                                      ...) {

  check_unused(...)

  args <- list(k = as.integer(k), center = center, permute = permute)
  args$end_times <- np_or_null(end_times)
  args$rewards <- rewards

  curves <- do.call(x$`_plot_accumulation_data`, args)

  line_plot(
    curve_frame(curves),
    xlab = curves$xlabel,
    ylab = curves$ylabel,
    title = if (is.null(title)) curves$title else title,
    legend = curves$legend_title
  )
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
#' @param t Times at which to evaluate the trajectories, `NULL` for the default.
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
#' pg <- load_phasegen()
#'
#' inf <- pg$Inference(
#'   bounds = list(t = c(0, 4), Ne = c(0.1, 1)),
#'   observation = pg$SFS(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652)),
#'   coal = function(t, Ne) pg$Coalescent(
#'     n = 10,
#'     demography = pg$Demography(events = c(
#'       pg$PopSizeChange(pop = "pop_0", time = 0, size = 1),
#'       pg$PopSizeChange(pop = "pop_0", time = t, size = Ne)
#'     ))
#'   ),
#'   loss = function(coal, obs) pg$PoissonLikelihood()$compute(
#'     observed = obs$normalize()$polymorphic,
#'     modelled = coal$sfs$mean$normalize()$polymorphic
#'   ),
#'   resample = function(sfs, rng) sfs$resample(seed = rng),
#'   do_bootstrap = TRUE,
#'   n_bootstraps = 10L,
#'   seed = 42L
#' )
#'
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

  check_unused(...)
  which <- match.arg(which)

  if (which == "bootstraps") {
    return(plot_bootstraps(x, match.arg(kind), bins, title))
  }

  result <- x$`_plot_demography_data`(np_or_null(t), if (which == "demography") "all" else which, include_bootstraps)
  curves <- result[[1]]
  data <- curve_frame(curves)
  data$series <- factor(data$series, levels = unique(data$series))
  under <- NULL

  if (length(result[[2]]) > 0) {

    boots <- do.call(rbind, Map(function(b, i) {
      frame <- curve_frame(b)
      frame$curve <- paste(i, frame$curve)
      frame
    }, result[[2]], seq_along(result[[2]])))

    boots$series <- factor(boots$series, levels = levels(data$series))
    under <- ggplot2::geom_step(data = boots, direction = "hv", alpha = 0.3)
  }

  line_plot(data, xlab = curves$xlabel, ylab = curves$ylabel, title = if (is.null(title)) curves$title else title,
            step = TRUE, under = under)
}


# histograms or density estimates of the bootstrapped parameters, one panel per parameter
plot_bootstraps <- function(x, kind, bins, title) {

  names <- as.character(unlist(x$param_names))
  values <- matrix(as.numeric(x$`_bootstrap_values`), ncol = length(names))

  if (nrow(values) == 0) {
    stop("No bootstraps available.", call. = FALSE)
  }

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
