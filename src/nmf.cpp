#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>

// generate a random uint32 given indices i and j and some state
inline uint32_t rand_uint32(const uint32_t state, const uint32_t i, const uint32_t j) {
    // generate a unique hash of i and j, using (max(i, j))(max(i, j) + 1) / 2 + min(i, j)
    // https://math.stackexchange.com/questions/882877/produce-unique-number-given-two-integers
    // credit to user @JimmyK4542, and whoever published the original intuition
    // also add 1 to i and j to avoid issues where i == 0 || j == 0
    // transform to uint64_t to avoid issues with overflow during multiplication
    uint64_t ij = (i + 1) * (i + 2) / 2 + j + 1;

    // adapted from xorshift64, Marsaglia
    // https://en.wikipedia.org/wiki/Xorshift
    ij ^= ij << 13 | (i << 17);
    ij ^= ij >> 7 | (j << 5);
    ij ^= ij << 17;

    // adapted from xorshift128+
    // https://xoshiro.di.unimi.it/xorshift128plus.c
    uint64_t s = state ^ ij;
    s ^= s << 23;
    s = s ^ ij ^ (s >> 18) ^ (ij >> 5);
    return (uint32_t)((s + ij));
}

// generate a random value in the uniform distribution [0, 1]
inline double rand_unif(const uint32_t state, const uint32_t i, const uint32_t j) {
    double x = (double)rand_uint32(state, i, j) / UINT32_MAX;
    return x - std::floor(x);
}

// generate a sparse matrix
//[[Rcpp::export]]
Eigen::SparseMatrix<double> rand_spmat(const uint32_t nrow, const uint32_t ncol, const uint32_t inv_density, const uint32_t seed) {
    Eigen::SparseMatrix<double> mat(nrow, ncol);
    mat.reserve(Eigen::VectorXi::Constant(ncol, nrow / (inv_density - 1)));
    for (uint32_t j = 0; j < ncol; ++j) {
        for (uint32_t i = 0; i < nrow; ++i) {
            if (rand_uint32(seed, i, j) % inv_density == 0) {
                mat.insert(i, j) = rand_unif(seed, i, j);
            }
        }
    }
    mat.makeCompressed();
    return mat;
}

// generate a dense matrix
//[[Rcpp::export]]
Eigen::MatrixXd rand_mat(const uint32_t nrow, const uint32_t ncol, const uint32_t seed) {
    Eigen::MatrixXd mat(nrow, ncol);
    for (uint32_t j = 0; j < ncol; ++j) {
        for (uint32_t i = 0; i < nrow; ++i) {
            mat(i, j) = rand_unif(seed, i, j);
        }
    }
    return mat;
}

// NMF HELPER FUNCTIONS
// Pearson correlation between two matrices (used for determining convergence)
inline double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
    double x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        x_i = (*(x.data() + i));
        y_i = (*(y.data() + i));
        sum_x += x_i;
        sum_y += y_i;
        sum_xy += x_i * y_i;
        sum_x2 += x_i * x_i;
        sum_y2 += y_i * y_i;
    }
    return 1 - (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
}

// fast symmetric matrix multiplication, A * A.transpose()
// see https://stackoverflow.com/questions/72100483/matrix-multiplication-of-an-eigen-matrix-for-a-subset-of-columns
Eigen::MatrixXd AAt(const Eigen::MatrixXd& A) {
    Eigen::MatrixXd AAt = Eigen::MatrixXd::Zero(A.rows(), A.rows());
    AAt.selfadjointView<Eigen::Lower>().rankUpdate(A);
    AAt.triangularView<Eigen::Upper>() = AAt.transpose();
    AAt.diagonal().array() += 1e-15;  // for numerical stability during coordinate descent NNLS
    return AAt;
}

// scale rows in w (or h) to sum to 1 and put previous rowsums in d
void scale(Eigen::MatrixXd& w, Eigen::VectorXd& d) {
    d = w.rowwise().sum();
    d.array() += 1e-15;
    for (size_t i = 0; i < w.rows(); ++i)
        for (size_t j = 0; j < w.cols(); ++j)
            w(i, j) /= d(i);
};

// NNLS SOLVER FOR SYSTEMS IN THE FORM OF ax=b
// optimized and modified from github.com/linxihui/NNLM "c_nnls" function
inline void nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const size_t sample) {
    double tol = 1;
    for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8; ++it) {
        tol = 0;
        for (size_t i = 0; i < h.rows(); ++i) {
            double diff = b(i) / a(i, i);
            if (-diff > h(i, sample)) {
                if (h(i, sample) != 0) {
                    b -= a.col(i) * -h(i, sample);
                    tol = 1;
                    h(i, sample) = 0;
                }
            } else if (diff != 0) {
                h(i, sample) += diff;
                b -= a.col(i) * diff;
                tol += std::abs(diff / (h(i, sample) + 1e-15));
            }
        }
    }
}

// NMF UPDATE FUNCTIONS
// update h given A and w
void predict(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const double L1) {
    Eigen::MatrixXd a = AAt(w);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < h.cols(); ++i) {
        Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());

        // BOTTLENECK OPERATION
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
            b += it.value() * w.col(it.row());
        // END BOTTLENECK OPERATION
        b.array() -= L1;
        nnls(a, b, h, i);
    }
}



void fit() {
    if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");

    // alternating least squares updates
    for (; iter_ < maxit; ++iter_) {
        Eigen::MatrixXd w_it = w;
        predictH();  // update "h"
        scaleH();
        predictW();  // update "w"
        scaleW();
        tol_ = cor(w, w_it);  // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        if (tol_ < tol) break;
        Rcpp::checkUserInterrupt();
    }

    if (tol_ > tol && iter_ == maxit && verbose)
        Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);

    if (sort_model) sortByDiagonal();
}

// NMF FUNCTION
//[[Rcpp::export]]
Rcpp::List c_nmf(const Eigen::SparseMatrix<double> A, const double tol, const uint16_t maxit, const bool verbose,
                 const double L1, Eigen::MatrixXd w) {
    const Eigen::SparseMatrix<double> At = A.transpose();
    if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
    if (w.rows() == A.rows()) w = w.transpose();
    if (w.cols() != A.rows()) Rcpp::stop("dimensions of A and w are incompatible");
    Eigen::MatrixXd h(w.rows(), A.cols());
    Eigen::VectorXd d(w.rows());
    double tol_ = 1;
    for (size_t iter_ = 0; iter_ < maxit && tol_ > tol; ++iter_) {
        Eigen::MatrixXd w_it = w;

        // update h, scale h, update w, scale w
        predict(A, w, h, L1);
        scale(h, d);
        predict(At, h, w, L1);
        scale(w, d);

        // calculate tolerance of the model fit to detect convergence
        tol_ = cor(w, w_it);  // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        Rcpp::checkUserInterrupt();
    }

    return Rcpp::List::create(Rcpp::Named("w") = w, Rcpp::Named("d") = d, Rcpp::Named("h") = h);
}

// NMF FUNCTION
//[[Rcpp::export]]
size_t c_nmf_rand(const uint32_t seed, const uint32_t nrow, const uint32_t ncol, const uint32_t k, const uint16_t maxit) {
    // use inv_density = 20 to generate a 95% sparse matrix
    Eigen::SparseMatrix<double> A = rand_spmat(nrow, ncol, 20, seed);
    Eigen::MatrixXd w = rand_mat(k, nrow, seed);

    // time this for n different seeds
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    c_nmf(A, 1e-20, maxit, false, 0, w);
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    size_t res = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    return res;
}

//' Automatic benchmarking using random matrices
//'
//' @export
//[[Rcpp::export]]
std::vector<size_t> run_benchmarking() {
    std::vector<uint32_t> seeds = {182274, 10483};
    std::vector<uint32_t> ranks = {5, 10, 20, 30, 40, 50};
    std::vector<uint32_t> nrows = {1000, 10000, 25000, 50000, 100000, 200000};
    std::vector<uint32_t> ncols = {1000, 10000, 25000, 50000, 100000};
    std::vector<size_t> times;
    for (auto seed : seeds) {
        for (auto rank : ranks) {
            for (auto nrow : nrows) {
                for (auto ncol : ncols) {
                  Rcpp::checkUserInterrupt();
                    Rprintf("seed: %8i; rank: %2i, rows: %8i, cols: %8i\n", seed, rank, nrow, ncol);
                    size_t time = c_nmf_rand(seed, nrow, ncol, rank, 100);
                    times.push_back(time);
                }
            }
        }
    }
    return times;
}


void predict(Eigen::MatrixXd& A, Rcpp::SparseMatrix& m, const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h, const double L1, const double L2,
             const unsigned int threads, const bool mask) {
    if (!mask) {
        // GENERAL RANK IMPLEMENTATION
        Eigen::MatrixXd a = w * w.transpose();
        a.diagonal().array() += TINY_NUM + L2;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            // calculate right-hand side of system of equations, "b"
            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());

            b += w * A.col(i);
            // subtract L1 penalty from "b"
            if (L1 != 0) b.array() -= L1;

            c_nnls(a, b, h, i);
        }
    } else if (mask) {
        Eigen::MatrixXd a = w * w.transpose();
        h.setZero();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
        for (unsigned int i = 0; i < h.cols(); ++i) {
            // subtract contribution of masked rows from "a"
            std::vector<unsigned int> masked_rows_ = m.InnerIndices(i);
            Eigen::VectorXi masked_rows(masked_rows_.size());
            for (unsigned int j = 0; j < masked_rows.size(); ++j)
                masked_rows(j) = (int)masked_rows_[j];
            Eigen::MatrixXd w_ = submat(w, masked_rows);
            Eigen::MatrixXd a_ = w_ * w_.transpose();
            a_ = a - a_;
            a_.diagonal().array() += TINY_NUM + L2;

            // calculate "b" for all non-masked rows
            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());

            b += w * A.col(i);
            // subtract contributions of masked rows from "b"
            for (unsigned int it = 0; it < masked_rows_.size(); ++it)
                b -= A(masked_rows_[it], i) * w.col(masked_rows_[it]);
            if (L1 != 0) b.array() -= L1;


            // solve system with least squares
            c_nnls(a_, b, h, i);
        }
    }
}

#endif

// solve for 'h' given sparse 'A' in 'A = wh'
void predict(Rcpp::SparseMatrix A, Rcpp::SparseMatrix mask_A, Rcpp::SparseMatrix& mask_h, const Eigen::MatrixXd& w,
             Eigen::MatrixXd& h, const double L1, const double L2, const int threads, const bool mask_zeros,
             const bool masking_A, const bool masking_h, const double upper_bound) {
    // set upper_bound = 0 to not impose an upper bound
    if (!mask_zeros) {
        // calculate "a"
        //  * calculate "a" for updates of all columns of "h"
        //  * if masking is applied to "A", we will subtract away the contributions of masked
        //       values in each column update
        Eigen::MatrixXd a = w * w.transpose();
        a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
        for (int i = 0; i < h.cols(); ++i) {
            // if there are no nonzeros in this column of "A", no need to solve anything
            if (A.p[i] == A.p[i + 1]) continue;

            // find the number of masked values in "A.col(i)"
            int num_masked = 0;
            if (masking_A)
                num_masked = mask_A.p[i + 1] - mask_A.p[i];

            Eigen::MatrixXd a_i;

            // calculate "b"
            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
            if (num_masked == 0) {
                // calculate "b" without masking on "A"
                for (Rcpp::SparseMatrix::InnerIterator it(A, i); it; ++it)
                    b += it.value() * w.col(it.row());
            } else {
                // calculate "b" with weighted masking on "A"
                //  * traverse both A.col(i) and mask_A.col(i) similar to a boost ForwardTraversalIterator
                Rcpp::SparseMatrix::InnerIterator it_A(A, i), it_mask(mask_A, i);
                while (it_A) {
                    if (!it_mask || it_A.row() < it_mask.row()) {
                        b += it_A.value() * w.col(it_A.row());
                        ++it_A;
                    } else if (it_mask && it_A.row() == it_mask.row()) {
                        if (it_mask.value() < 1)
                            b += ((it_A.value() * (1 - it_mask.value())) * w.col(it_A.row()));
                        ++it_mask;
                        ++it_A;
                    } else if (it_mask) {
                        ++it_mask;
                    }
                }
                // if masking values in A.col(i), subtract contributions of masked indices from "a"
                //  * we only need to consider columns in "w" that correspond to non-zero rows in "A" because
                //      this code block does not consider the masked_zeros case
                Eigen::MatrixXd w_(w.rows(), num_masked);
                int j = 0;
                for (Rcpp::SparseMatrix::InnerIterator it(mask_A, i); it; ++it, ++j)
                    w_.col(j) = w.col(it.row()) * it.value();
                Eigen::MatrixXd a_ = w_ * w_.transpose();
                a_i = a - a_;
            }

            // apply L1 penalty on "b"
            if (L1 != 0) b.array() -= L1;

            // apply masking on "h"
            if (masking_h) {
                Rcpp::NumericVector mask_h_i = mask_h.col(i);
                for (int k = 0; k < b.size(); ++k)
                    b[k] *= mask_h_i[k];
            }

            // solve nnls equations
            if (upper_bound > 0) {
                (num_masked == 0) ? c_bnnls(a, b, h, i, upper_bound) : c_bnnls(a_i, b, h, i, upper_bound);
            } else {
                (num_masked == 0) ? c_nnls(a, b, h, i) : c_nnls(a_i, b, h, i);
            }
        }
    } else {  // mask_zeros = true
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
        for (int i = 0; i < h.cols(); ++i) {
            if (A.p[i] == A.p[i + 1]) continue;

            int num_masked = 0;
            if (masking_A)
                num_masked = mask_A.p[i + 1] - mask_A.p[i];

            Eigen::VectorXi nnz(A.p[i + 1] - A.p[i]);
            for (int ind = A.p[i], j = 0; j < nnz.size(); ++ind, ++j)
                nnz(j) = A.i[ind];
            Eigen::MatrixXd w_ = submat(w, nnz);

            Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
            if (num_masked == 0) {
                for (Rcpp::SparseMatrix::InnerIterator it(A, i); it; ++it)
                    b += it.value() * w.col(it.row());
            } else {
                // subset "w" at non-masked indices in A.col(i) to calculate "a"
                Rcpp::SparseMatrix::InnerIterator it_mask(mask_A, i), it_A(A, i);
                int j = 0;
                while (it_mask && it_A) {
                    if (it_mask.row() == it_A.row()) {
                        w_.col(j) *= (1 - it_mask.value());
                        ++it_mask;
                        ++it_A;
                        ++j;
                    } else if (it_mask.row() < it_A.row()) {
                        ++it_mask;
                    } else {
                        ++it_A;
                    }
                }

                // calculate "b" with masking on "A"
                Rcpp::SparseMatrix::InnerIterator it_mask2(mask_A, i), it_A2(A, i);
                while (it_A) {
                    if (!it_mask2 || it_A2.row() < it_mask2.row()) {
                        b += it_A2.value() * w.col(it_A2.row());
                        ++it_A2;
                    } else if (it_mask2 && it_A2.row() == it_mask2.row()) {
                        if (it_mask2.value() < 1)
                            b += ((it_A2.value() * (1 - it_mask2.value())) * w.col(it_A2.row()));
                        ++it_mask2;
                        ++it_A2;
                    } else if (it_mask2) {
                        ++it_mask2;
                    }
                }
            }
            Eigen::MatrixXd a = w_ * w_.transpose();

            if (L1 != 0) b.array() -= L1;
            a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;
            if (masking_h) {
                Rcpp::NumericVector mask_h_i = mask_h.col(i);
                for (int k = 0; k < b.size(); ++k)
                    b[k] *= mask_h_i[k];
            }
            if (upper_bound > 0) {
                c_bnnls(a, b, h, i, upper_bound);
            } else {
                c_nnls(a, b, h, i);
            }
        }
    }
}
