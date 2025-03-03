#' Validate Imputed Values Using Non-negative Matrix Factorization (NMF)
#'
#' This function evaluates the performance of Non-negative Matrix Factorization (NMF) in imputing missing values
#' by generating a validation mask, training an NMF model, and assessing reconstruction accuracy.
#'
#' @param dataList A list of sparse matrices (`dgCMatrix`) to be stacked and used for imputation validation.
#' @param k An integer specifying the rank (number of components) for NMF decomposition.
#' @param dropout A numeric value (between 0 and 1) specifying the proportion of values to randomly mask for validation.
#' @param seed An optional numeric value for reproducibility in NMF (default: NULL).
#' @return The trained NMF model (`RcppML::nmf` object).
#' @details
#' This function:
#' 1. Creates a **validation mask** by randomly masking a proportion (`dropout`) of the input data.
#' 2. Stacks the input sparse matrices (`dataList`) into a single matrix.
#' 3. Replaces the masked values with `NaN` in the training data.
#' 4. Runs **Non-negative Matrix Factorization (NMF)** using `RcppML::nmf()` on the training data.
#'
#' The commented-out `evaluate` function in RcppML can be used to compare the NMF reconstruction
#' against the original data (`stackedData`), evaluating its performance on masked entries.
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
#' # Validate imputed values using NMF with k=2 and 20% dropout
#' model <- validateImputedValues(dataList, k = 2, dropout = 0.2, seed = 42)
#'
#' print(model)
#'
#' @import Matrix
#' @import RcppML
#' @export
validateImputedValues = function(dataList,k,dropout,seed=NULL){
  validationMask = getRandomMask(dataList,dropout,123)
  stackedData = stackMatrix(dataList)
  trainData = replaceValues(stackedData,validationMask,NaN)

  mod = RcppML::nmf(trainData, k = k, mask='NA', seed = seed, tol = 1e-3)

  #RcppML::evaluate(mod, stackedData, mask = validationMask, missing_only = TRUE)
}

replaceValues = function(data,indices,value=NaN){
  trainData = data
  trainData[indices] = value
  return(trainData)
}
