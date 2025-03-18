#' Get Missing Indices in Data Lists
#'
#' This function identifies missing column indices for each dataset within a list.
#' It determines which columns from the full set of unique column names are missing
#' in each dataset and returns their row and column indices.
#'
#' @param dataList A list of data frames with potentially different column names.
#'
#' @return A list with two elements:
#' \describe{
#'   \item{rowCoord}{A list of row index ranges corresponding to each data frame in `dataList`.}
#'   \item{colCoord}{A list of column indices indicating the missing columns for each data frame.}
#' }
#'
#' @examples
#' # Example usage
#' df1 <- data.frame(A = 1:3, B = 4:6)
#' df2 <- data.frame(B = 7:9, C = 10:12)
#' dataList <- list(df1, df2)
#' getMissingIndices(dataList)
#'
#' @export
getMissingIndices = function(dataList){
  all_cells = getUniqueCells(dataList)

  row_coord = vector("list",length(dataList))
  col_coord = vector("list",length(dataList))
  startRow = 1
  for (i in 1:length(dataList)){
    nRow = nrow(dataList[[i]])
    endRow = startRow+nrow(dataList[[i]])-1

    missing_cells <- setdiff(all_cells, colnames(dataList[[i]]))  # Find missing columns
    row_coord[[i]] = startRow:endRow
    col_coord[[i]] = which(all_cells %in% missing_cells)

    startRow = startRow+nRow
  }
  return(list(
    rowCoord = row_coord,
    colCoord = col_coord)
  )
}
