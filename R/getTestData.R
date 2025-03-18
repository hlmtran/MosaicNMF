#' Generate Test Data for Sparse Matrices
#'
#' This function creates a list of sparse matrices (`K27`, `CUTAC`, `K4`) with randomly generated binary values (0 or 1).
#' The matrices simulate different datasets with overlapping and unique column names.
#'
#' @return A list of sparse matrices:
#' \describe{
#'   \item{K27}{A 10-row matrix with 5 columns labeled as `Cell1` to `Cell5`.}
#'   \item{CUTAC}{A 10-row matrix with 5 columns labeled as `Cell2`, `Cell3`, `Cell6`, `Cell7`, `Cell8`.}
#'   \item{K4}{A 10-row matrix with 5 columns labeled as `Cell3`, `Cell4`, `Cell7`, `Cell9`, `Cell10`.}
#' }
#'
#' @details
#' - Each matrix has **10 rows**, with row names prefixed by `"K27_"`, `"CUTAC_"`, or `"K4_"`, respectively.
#' - Column names simulate cell identifiers, demonstrating a scenario where datasets share some cells but not others.
#' - The function sets a random seed (`set.seed(123)`) to ensure reproducibility.
#'
#' @examples
#' testData <- getTestData()
#' str(testData)  # Check the structure of the output
#'
#' # Accessing individual matrices
#' testData[[1]]  # K27 matrix
#' testData[[2]]  # CUTAC matrix
#' testData[[3]]  # K4 matrix
#'
#' @export
getTestData= function(){
  set.seed(123)

  K27 <- Matrix::Matrix(sample(0:1, 50, replace=TRUE), nrow=10, sparse=TRUE)
  CUTAC <- Matrix::Matrix(sample(0:1, 50, replace=TRUE), nrow=10, sparse=TRUE)
  K4 <- Matrix::Matrix(sample(0:1, 50, replace=TRUE), nrow=10, sparse=TRUE)

  # Assign cell names (column names)
  colnames(K27) <- c("Cell1", "Cell2", "Cell3", "Cell4", "Cell5")
  colnames(CUTAC) <- c("Cell2", "Cell3", "Cell6", "Cell7", "Cell8")
  colnames(K4) <- c("Cell3", "Cell4", "Cell7", "Cell9", "Cell10")

  rownames(K27) = paste0("K27_",1:10)
  rownames(K4) = paste0("K4_",1:10)
  rownames(CUTAC) = paste0("CUTAC_",1:10)

  return(testList = c(K27=K27,CUTAC=CUTAC,K4=K4))
}
