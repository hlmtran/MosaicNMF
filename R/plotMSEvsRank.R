#' Plot Mean Squared Error (MSE) vs. Rank for NMF
#'
#' Computes and plots the Mean Squared Error (MSE) of Non-negative Matrix Factorization (NMF)
#' for different ranks to help evaluate the optimal rank for decomposition.
#'
#' @param dataset A numeric matrix (dense or sparse) representing the data to be factorized.
#' @param ranks A numeric vector specifying the different rank values to test.
#' @param mask A matrix of the same dimensions as `dataset` specifying missing values to be ignored (optional, default: NULL).
#' @param seed A numeric value specifying the random seed for reproducibility (optional, default: NULL).
#' @return A plot of MSE vs. rank.
#' @details This function applies NMF to the dataset for each rank in `ranks`, calculates the reconstruction MSE, and plots the results.
#' @examples
#' library(RcppML)
#' dataset <- matrix(runif(100), 10, 10)  # Generate a random dataset
#' ranks <- seq(2, 10, by = 2)  # Define ranks to test
#' plotMSEvsRank(dataset, ranks)
#' @import RcppML
#' @export
plotMSEvsRank = function(dataset,ranks,mask=NULL,seed=NULL){
  validation = list()
  for(rank in ranks){
    mod = RcppML::nmf(dataset, k = rank, mask = mask, seed = seed, tol = 1e-3)
    validation[[rank]] = RcppML::mse(mod@w,mod@d,mod@h,dataset)
  }

  plot(ranks,unlist(validation))
}
