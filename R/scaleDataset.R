#' Scale a Dataset to Sum to One
#'
#' This function scales all values in a matrix such that the sum of the matrix is equal to 1.
#'
#' @param dataset A numeric matrix (dense or sparse) to be scaled.
#' @return A matrix where all values are scaled so that their sum is exactly 1.
#' @details
#' This function normalizes the input matrix by dividing all elements by the total sum of the matrix.
#' The function also performs an assertion check to ensure the final sum is exactly 1.
#'
#' @examples
#' library(Matrix)
#'
#' # Create an example matrix
#' mat <- Matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, sparse = TRUE)
#'
#' # Scale the dataset
#' scaled_mat <- scaleDataSet(mat)
#'
#' print(sum(scaled_mat, na.rm = TRUE))  # Should print 1
#'
#' @import Matrix
#' @import assertthat
#' @export
scaleDataset = function(dataset){
  scaleFactor = sum(dataset,na.rm = TRUE)
  dataset = dataset/scaleFactor

  assertthat::assert_that(sum(dataset,na.rm=TRUE)==1,msg = "Sum of matrix not equal 1 after scaling.")
  return(dataset)
}
