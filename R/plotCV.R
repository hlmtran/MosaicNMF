#' Plot Cross-Validation Results for NMF
#'
#' @description
#' Visualizes the prediction error across different NMF ranks (`k`), grouped by replicate (`rep`), from cross-validation results.
#'
#' @param data A `data.frame` with columns `rep`, `k`, and `value`, where:
#' * `rep` indicates the replicate (factor or integer),
#' * `k` is the rank of factorization,
#' * `value` is the prediction error (typically mean squared error).
#'
#' @return A `ggplot` object showing line plots of prediction error vs. rank (`k`) for each replicate.
#'
#' @details
#' This function is useful for visualizing the results of non-negative matrix factorization (NMF) cross-validation, where the optimal rank minimizes prediction error.
#'
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   rep = as.factor(rep(1:3, each = 3)),
#'   k = rep(3:5, times = 3),
#'   value = runif(9, 0, 0.05)
#' )
#' plotCV(df)
#' }
#'
#' @import ggplot2
#' @export

plotCV = function(data){
  plot = ggplot(data, aes(x = factor(k), y = value, color = rep, group = rep)) +
    geom_line() +
    geom_point() +
    labs(
      title = "Cross-Validation Error by NMF Rank",
      x = "Rank (k)",
      y = "Prediction Error (MSE)",
      color = "Replicate"
    )
  return(plot)
}
