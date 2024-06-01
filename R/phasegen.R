
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
  
  # Check if phasegen is installed
  installed <- reticulate::py_module_available("phasegen")
  
  return(installed)
}


#' Install the `phasegen` Python module
#'
#' This function checks if the `phasegen` Python module is available.
#' If not, or if the `force` argument is TRUE, it installs it via pip.
#' If the `silent` argument is set to TRUE, the function will not output a 
#' message when the module is already installed.
#'
#' @param version A character string specifying the version of the `phasegen` module
#'        to install. Default is `NULL` which will install the latest version.
#' @param force Logical, if `TRUE` it will force the reinstallation of the `phasegen` module
#'        even if it's already available. Default is `FALSE`.
#' @param silent Logical, if `TRUE` it will suppress the message about `phasegen` being
#'        already installed. Default is `FALSE`.
#'
#' @return Invisible `NULL`.
#' 
#' @examples
#' \dontrun{
#' install_phasegen()  # Installs the latest version of phasegen
#' install_phasegen("1.1.7")  # Installs version 1.1.7 of phasegen
#' install_phasegen(force = TRUE)  # Reinstalls the phasegen module
#' }
#' 
#' @export
install_phasegen <- function(version = NULL, force = FALSE, silent = FALSE, python_version = '3.11') {
  
  # Create the package string with the version if specified
  package_name <- "phasegen"
  if (!is.null(version)) {
    package_name <- paste0(package_name, "==", version)
  }
  
  # Check if phasegen is installed or if force is TRUE
  if (force || !phasegen_is_installed()) {
    reticulate::py_install(
      package_name, 
      method = "conda",
      pip = TRUE,
      python_version = python_version,
      version = version, 
      ignore_installed = TRUE
   )
  } else {
    if (!silent) {
      message("The 'phasegen' Python module is already installed.")
    }
  }
  
  invisible(NULL)
}

#' Load the phasegen library and associated visualization functions
#'
#' This function imports the Python package 'phasegen' using the reticulate package
#' and then configures it to work seamlessly with R, overriding some of the default
#' visualization functions with custom R-based ones. This function also ensures
#' that required R libraries are loaded for visualization.
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
  
  # use super assignment to update global reference to phasegen
  pg <- reticulate::import("phasegen")
}
