% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/stackMatrix.R
\name{align_sparse_matrix}
\alias{align_sparse_matrix}
\title{Align a Sparse Matrix to a Common Set of Columns}
\usage{
align_sparse_matrix(mat, all_cells, missingValue = NaN)
}
\arguments{
\item{mat}{A sparse matrix (\code{dgCMatrix}) that needs alignment.}

\item{all_cells}{A character vector specifying the full set of column names to align with.}

\item{missingValue}{A numeric or logical value to fill missing columns (default: \code{NaN}).}
}
\value{
A sparse matrix (\code{dgCMatrix}) with columns aligned to \code{all_cells}, with missing columns filled with \code{missingValue}.
}
\description{
Ensures a sparse matrix (\code{dgCMatrix}) has all specified columns by adding missing columns filled with a specified value.
The function also reorders columns to match the given reference order.
}
\details{
This function ensures that \code{mat} contains all columns in \code{all_cells}. Any missing columns are added and filled
with \code{missingValue} (default: \code{NaN}). The resulting matrix is also reordered to match \code{all_cells}.
}
\examples{
library(Matrix)

# Create an example sparse matrix
mat <- Matrix(c(1, 0, 2, 0, 3, 4), nrow = 2, sparse = TRUE)
colnames(mat) <- c("A", "B", "C")

# Define full set of columns
all_cells <- c("A", "B", "C", "D", "E")

# Align matrix with missing columns filled with NaN
aligned_mat <- align_sparse_matrix(mat, all_cells, missingValue = NaN)

print(aligned_mat)

}
