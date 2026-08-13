
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
  
  pg <- reticulate::import("phasegen")
}
