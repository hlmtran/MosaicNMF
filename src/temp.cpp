#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <algorithm>
//#include <Eigen/Sparse>

using namespace Rcpp;
// Explicitly define library usage
//typedef Eigen::SparseMatrix<bool> SparseBoolMatrix;
//typedef Eigen::Triplet<bool> Triplet;

// [[Rcpp::export]]
std::vector<bool> getMaskVector(int ind, const std::string& dim,
                                const std::vector<std::vector<int>>& rows,
                                const std::vector<std::vector<int>>& cols,
                                int dimSize) {
  std::vector<bool> maskVector(dimSize, false); // Initialize mask vector with false

  const std::vector<std::vector<int>>& toCheck = (dim == "row") ? rows : cols;
  const std::vector<std::vector<int>>& toMask = (dim == "row") ? cols : rows;

  for (size_t i = 0; i < toCheck.size(); ++i) {
    if (std::find(toCheck[i].begin(), toCheck[i].end(), ind) != toCheck[i].end()) {
      for (int idx : toMask[i]) {
        if (idx >= 0 && idx < dimSize) {
          maskVector[idx] = true;
        }
      }
    }
  }
  return maskVector;
}
/*
SparseBoolMatrix getMaskMatrix(const std::vector<std::vector<int>>& missingRows,
                               const std::vector<std::vector<int>>& missingColumns,
                               const std::string& dim, int numRow, int numCol) {
  SparseBoolMatrix mask(numRow, numCol); // Initialize a sparse matrix
  std::vector<Triplet> triplets; // Store non-zero elements

  if (dim == "row") {
    for (int row = 0; row < numRow; ++row) {
      std::vector<bool> maskVector = getMaskVector(row + 1, dim, missingRows, missingColumns, numCol);
      for (int col = 0; col < numCol; ++col) {
        if (maskVector[col]) {
          triplets.emplace_back(row, col, true);
        }
      }
    }
  } else {
    for (int col = 0; col < numCol; ++col) {
      std::vector<bool> maskVector = getMaskVector(col + 1, dim, missingRows, missingColumns, numRow);
      for (int row = 0; row < numRow; ++row) {
        if (maskVector[row]) {
          triplets.emplace_back(row, col, true);
        }
      }
    }
  }

  mask.setFromTriplets(triplets.begin(), triplets.end()); // Construct sparse matrix
  return mask;
}

*/
