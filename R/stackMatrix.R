#' Stack and Align Sparse Matrices
#'
#' Combines a list of sparse matrices by stacking them row-wise while aligning their columns based on unique cell names.
#' Missing values in the alignment process are filled with `NaN`.
#'
#' @param dataList A list of sparse matrices (each of class `dgCMatrix`) to be stacked.
#' @return A sparse numeric matrix (`dgCMatrix`) with rows from all matrices in `dataList` and columns corresponding to unique cell names.
#' Missing values are set to `NaN`.
#' @details
#' This function ensures that all matrices in `dataList` are aligned to the same column names before stacking them.
#' It creates a sparse matrix large enough to contain all rows from `dataList` while maintaining sparsity.
#'
#' Row names are preserved and updated accordingly.
#'
#' @examples
#' library(Matrix)
#'
#' # Create example sparse matrices
#' mat1 <- Matrix(sample(c(0, 1), 20, replace = TRUE), nrow = 4, sparse = TRUE)
#' colnames(mat1) <- c("A", "B", "C", "D", "E")
#' rownames(mat1) <- paste0("R", 1:4)
#'
#' mat2 <- Matrix(sample(c(0, 1), 15, replace = TRUE), nrow = 3, sparse = TRUE)
#' colnames(mat2) <- c("A", "B", "C", "D", "F")
#' rownames(mat2) <- paste0("R", 5:7)
#'
#' dataList <- list(mat1, mat2)
#'
#' # Stack matrices while aligning columns
#' stackedMat <- stackMatrix(dataList)
#'
#' print(stackedMat)
#'
#' @import Matrix
#' @export
stackMatrix = function(dataList){
  uniqueCells = getUniqueCells(dataList)

  alignedMat = Matrix::Matrix(0,getTotalRows(dataList),length(uniqueCells),sparse = TRUE)
  colnames(alignedMat) = uniqueCells
  rownames(alignedMat) = paste0("row",1:nrow(alignedMat)) #need to have row names to update row names

  startRow = 1
  for (i in 1:length(dataList)){
    endRow = startRow+nrow(dataList[[i]])-1
    alignedMat[startRow:endRow,] = align_sparse_matrix(dataList[[i]],uniqueCells,NaN)
    rownames(alignedMat)[startRow:endRow] = rownames(dataList[[i]])
    startRow = startRow+nrow(dataList[[i]])
  }
  return(alignedMat)
}

getUniqueCells = function(dataList){
  all_cells = lapply(dataList,colnames) |> unlist() |> unique()
  return(all_cells)
}
getTotalRows = function(dataList){
  tot = lapply(dataList,nrow) |> unlist() |> sum()
  return(tot)
}

#' Align a Sparse Matrix to a Common Set of Columns
#'
#' Ensures a sparse matrix (`dgCMatrix`) has all specified columns by adding missing columns filled with a specified value.
#' The function also reorders columns to match the given reference order.
#'
#' @param mat A sparse matrix (`dgCMatrix`) that needs alignment.
#' @param all_cells A character vector specifying the full set of column names to align with.
#' @param missingValue A numeric or logical value to fill missing columns (default: `NaN`).
#' @return A sparse matrix (`dgCMatrix`) with columns aligned to `all_cells`, with missing columns filled with `missingValue`.
#' @details
#' This function ensures that `mat` contains all columns in `all_cells`. Any missing columns are added and filled
#' with `missingValue` (default: `NaN`). The resulting matrix is also reordered to match `all_cells`.
#'
#' @examples
#' library(Matrix)
#'
#' # Create an example sparse matrix
#' mat <- Matrix(c(1, 0, 2, 0, 3, 4), nrow = 2, sparse = TRUE)
#' colnames(mat) <- c("A", "B", "C")
#'
#' # Define full set of columns
#' all_cells <- c("A", "B", "C", "D", "E")
#'
#' # Align matrix with missing columns filled with NaN
#' aligned_mat <- align_sparse_matrix(mat, all_cells, missingValue = NaN)
#'
#' print(aligned_mat)
#'
#' @import Matrix
#' @export
align_sparse_matrix <- function(mat, all_cells, missingValue = NaN) {
  missing_cells <- setdiff(all_cells, colnames(mat))  # Find missing columns
  extra_cols <- Matrix::Matrix(missingValue, nrow(mat), length(missing_cells), sparse=TRUE)  # Create sparse zero columns
  colnames(extra_cols) <- missing_cells
  mat_aligned <- cbind(mat, extra_cols)  # Add missing columns
  mat_aligned <- mat_aligned[, all_cells, drop=FALSE]  # Reorder columns
  return(mat_aligned)
}

