#' Generate a Random Mask for a List of Sparse Matrices
#'
#' Creates a logical sparse matrix (`lgCMatrix`) with `TRUE/FALSE` values,
#' aligning multiple matrices based on their unique cell names and randomly masking entries.
#'
#' @param dataList A list of sparse matrices (each of class `dgCMatrix`) to be aligned.
#' @param ratio A numeric value (between 0 and 1) specifying the proportion of `TRUE` values (non-masked elements).
#' @param seed An optional numeric value to set the random seed for reproducibility (default: NULL).
#' @return A sparse logical matrix (`lgCMatrix`) with `TRUE` (non-masked) and `FALSE` (masked) values.
#' @details
#' This function aligns a list of sparse matrices by mapping them to a shared set of unique cells (column names).
#' It then applies a random mask based on the specified `ratio`, ensuring row names are properly assigned.
#'
#' @examples
#' library(Matrix)
#'
#' # Create example sparse matrices
#' mat1 <- Matrix(sample(c(TRUE, FALSE), 20, replace = TRUE), nrow = 4, sparse = TRUE)
#' colnames(mat1) <- c("A", "B", "C", "D", "E")
#' rownames(mat1) <- paste0("R", 1:4)
#'
#' mat2 <- Matrix(sample(c(TRUE, FALSE), 15, replace = TRUE), nrow = 3, sparse = TRUE)
#' colnames(mat2) <- c("A", "B", "C", "D", "F")
#' rownames(mat2) <- paste0("R", 5:7)
#'
#' dataList <- list(mat1, mat2)
#'
#' # Generate a random mask with 20% sparsity
#' mask <- getRandomMask(dataList, ratio = 0.2, seed = 42)
#'
#' print(mask)
#'
#' @import Matrix
#' @export
getRandomMask = function(dataList,ratio,seed=NULL){
  uniqueCells = getUniqueCells(dataList)

  alignedMat = Matrix::Matrix(FALSE,getTotalRows(dataList),length(uniqueCells),sparse = TRUE)
  colnames(alignedMat) = uniqueCells
  rownames(alignedMat) = paste0("row",1:nrow(alignedMat)) #need to have row names to update row names

  startRow = 1
  for (i in 1:length(dataList)){
    endRow = startRow+nrow(dataList[[i]])-1

    alignedMat[startRow:endRow,] = align_sparse_matrix(
      mat = getRandomLogical(dataList[[i]],ratio,seed),
      all_cells = uniqueCells,
      missingValue = FALSE)

    rownames(alignedMat)[startRow:endRow] = rownames(dataList[[i]])
    startRow = startRow+nrow(dataList[[i]])
  }
  return(alignedMat)
}

getRandomLogical = function(dataset,ratio,seed=null){
  if (!is.null(seed)){
    set.seed(seed)
  }
  n_row = nrow(dataset)
  n_col = ncol(dataset)

  randMat = Matrix::rsparsematrix(n_row, n_col, density = ratio, rand.x = NULL)
  dim(randMat)
  rownames(randMat) = rownames(dataset)
  colnames(randMat) = colnames(dataset)
  return(randMat)
}

