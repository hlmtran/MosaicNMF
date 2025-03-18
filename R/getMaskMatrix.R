#' Generate a Mask Matrix Based on Missing Values
#'
#' This function generates a binary mask matrix indicating missing values in either rows or columns.
#'
#' @param missingRows A list where each element contains row indices that have missing values.
#' @param missingColumns A list where each element contains column indices corresponding to missing values.
#' @param dim A string indicating whether to apply masking by "row" or "column".
#' @param numRow An integer specifying the total number of rows in the matrix.
#' @param numCol An integer specifying the total number of columns in the matrix.
#'
#' @return A sparse logical matrix of dimensions (`numRow`, `numCol`) with `TRUE` indicating missing values.
#'
#' @examples
#' library(Matrix)
#' missingRows <- list(c(1, 3), c(2))
#' missingColumns <- list(c(2), c(3, 4))
#' getMaskMatrix(missingRows, missingColumns, dim = "row", numRow = 4, numCol = 4)
#'
#' @export
getMaskMatrix = function(missingRows,missingColumns,dim,numRow,numCol){
  mask = Matrix::Matrix(F,numRow,numCol)
  if (dim == "row"){
    for(row in 1:numRow){
      mask[row,] = getMaskVector(
        row,
        dim,
        missingRows,
        missingColumns,
        numCol)
    }
  }else{
    for(col in 1:numCol){
      mask[,col] = getMaskVector(
        col,
        dim,
        missingRows,
        missingColumns,
        numRow)
    }
  }
  return(mask)
}

#' Generate a Mask Vector Based on Missing Values
#'
#' This function generates a binary mask vector indicating missing values for a specific row or column.
#'
#' @param ind An integer specifying the row or column index being checked.
#' @param dim A string, either "row" or "column", indicating which dimension is being processed.
#' @param rows A list where each element contains row indices that have missing values.
#' @param cols A list where each element contains column indices corresponding to missing values.
#' @param dimSize An integer specifying the length of the mask vector (either the number of columns or rows).
#'
#' @return A logical vector of length `dimSize`, where `TRUE` indicates a missing value.
#'
#' @examples
#' rows <- list(c(1, 3), c(2))
#' cols <- list(c(2), c(3, 4))
#' getMaskVector(1, "row", rows, cols, 4)
#'
#' @export
getMaskVector = function(ind,dim,rows,cols,dimSize){
  maskVector = rep(F,dimSize)

  if(dim == "row"){
    toCheck = rows
    toMask = cols
  } else {
    toCheck = cols
    toMask = rows
  }
  for(data in 1:length(rows)){
    # print(data)
    # print(toCheck[[data]])
    if(ind %in% toCheck[[data]]){
      maskVector[toMask[[data]]] = T
    }
  }
  return(maskVector)
}
