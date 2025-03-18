#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

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
inline void nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const size_t sample, const double L2) {
    double tol = 1;
    for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8; ++it) {
        tol = 0;
        for (size_t i = 0; i < h.rows(); ++i) {
            double diff = b(i) / (a(i, i) + L2);
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
void predict_cd(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const double L1, const double L2) {
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
        nnls(a, b, h, i, L2);
    }
}

// NMF UPDATE FUNCTIONS
// update h given A and w
void predict_gd(const Eigen::SparseMatrix<double>& A,
                const Eigen::MatrixXd& w,
                Eigen::MatrixXd& h,
                const double L1,
                const double L2) {
    const size_t n = A.cols();
    const size_t m = A.rows();
    const size_t k = h.rows();

    // W^TW (k x k)
    Eigen::MatrixXd wtw = AAt(w);

    // A^TW (k x n)
    Eigen::MatrixXd atw = Eigen::MatrixXd::Zero(k, n);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
            atw.col(i) += it.value() * w.col(it.row());
        }
    }

    // update each row of H
    // TO DO:  atw becomes a vector
    for (size_t i = 0; i < k; ++i) {
        atw.row(i) -= h.transpose() * wtw.col(i);
        atw.row(i).array() /= wtw(i, i);
        atw.row(i) = (atw.row(i).array() < 0).select(0, atw.row(i));
        h.row(i) += atw.row(i);
    }
}

// NMF FUNCTION
// tied weights
// A is m x n
// w is k x m
//[[Rcpp::export]]
Rcpp::List autoencoder_nmf(const Eigen::SparseMatrix<double>& A,
                           Eigen::MatrixXd& w,
                           double learning_rate = 0.001,
                           size_t maxit = 100,
                           double tol = 1e-5,
                           bool verbose = true) {
    if (verbose) Rprintf("\n%4s | %8s | %8s\n-------------------------\n", "iter", "mae", "tol");
    size_t m = A.rows();
    size_t n = A.cols();
    size_t k = w.rows();

    Eigen::VectorXd err = Eigen::VectorXd::Zero(maxit);
    size_t iter = 0;
    double curr_tol = 1;
    for (; iter < maxit && curr_tol > tol; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            // FEED-FORWARD
            // encoder
            Eigen::VectorXd a1 = Eigen::VectorXd::Zero(k);
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
                a1 += it.value() * w.col(it.row());

            // a1 = (a1.array() < 0).select(0, a1);  // relu activation
            // if "A" is non-negative, there is no possibility for a1 to be negative

            // decoder
            Eigen::VectorXd a2(m);
            for (size_t j = 0; j < m; ++j)
                a2(j) = (w.col(j).array() * a1.array()).sum();

            // a2 = (a2.array() < 0).select(0, a2);  // relu activation

            // calculate error
            Eigen::VectorXd error = -a2;
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
                error(it.row()) += it.value();

            err(iter) += error.array().abs().sum() / m;  // mean absolute error

            // BACKPROPAGATE
            Eigen::ArrayXd a1_prime = (a1.array() > 0).select(1, a1);  // derivative of relu
            Eigen::ArrayXd a2_prime = (a2.array() > 0).select(1, a2);  // derivative of relu
            Eigen::ArrayXd a2_delta = error.array() * a2_prime;        // length m
            Eigen::ArrayXd a1_delta = Eigen::VectorXd::Zero(k);
            for (size_t j = 0; j < m; ++j)
                a1_delta += w.col(j).array() * a2_delta(j);
            a1_delta *= a1_prime;

            // update weights
            a1_delta *= learning_rate;
            a2_delta *= learning_rate;

            // update weights with encoder error
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
                for (size_t p = 0; p < k; ++p)
                    w(p, it.row()) += a1_delta(p) * it.value();

            // update weights with decoder error
            for (size_t p = 0; p < k; ++p)
                for (size_t q = 0; q < m; ++q)
                    w(p, q) += a1(p) * a2_delta(q);

            // apply non-negativity constraints
            w = (w.array() < 0).select(0, w);
        }
        err(iter) /= n;
        if (iter > 0) curr_tol = std::abs(err(iter) - err(iter - 1)) / (err(iter) + err(iter - 1));
        if (verbose) Rprintf("%4d | %8f | %8f\n", iter + 1, err(iter), curr_tol);
        Rcpp::checkUserInterrupt();
    }
    Eigen::MatrixXd h = w * A;
    w = w.transpose();
    Eigen::VectorXd d = Eigen::VectorXd::Zero(k);
    for (size_t j = 0; j < k; ++j) {
        double h_diag = h.row(j).array().sum();
        double w_diag = w.col(j).array().sum();
        d(j) = h_diag + w_diag;
        h.row(j).array() /= h_diag;
        w.col(j).array() /= w_diag;
    }
    err.conservativeResize(iter);
    return Rcpp::List::create(Rcpp::Named("w") = w, Rcpp::Named("d") = d, Rcpp::Named("h") = h, Rcpp::Named("error") = err);
}

// untied weights
// A is m x n
// w1 and w2 are k x m
//[[Rcpp::export]]
Rcpp::List autoencoder2_nmf(const Eigen::SparseMatrix<double>& A,
                            Eigen::MatrixXd& w1,
                            Eigen::MatrixXd& w2,
                            double learning_rate = 0.001,
                            size_t maxit = 100,
                            double tol = 1e-5,
                            bool verbose = true) {
    if (verbose) Rprintf("\n%4s | %8s | %8s\n-------------------------\n", "iter", "mae", "tol");
    size_t m = A.rows();
    size_t n = A.cols();
    size_t k = w1.rows();

    Eigen::VectorXd err = Eigen::VectorXd::Zero(maxit);
    size_t iter = 0;
    double curr_tol = 1;
    for (; iter < maxit && curr_tol > tol; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            // FEED-FORWARD
            // encoder
            Eigen::VectorXd a1 = Eigen::VectorXd::Zero(k);
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
                a1 += it.value() * w1.col(it.row());

            // a1 = (a1.array() < 0).select(0, a1);  // relu activation
            // if "A" is non-negative, there is no possibility for a1 to be negative

            // decoder
            Eigen::VectorXd a2(m);
            for (size_t j = 0; j < m; ++j)
                a2(j) = (w2.col(j).array() * a1.array()).sum();

            // a2 = (a2.array() < 0).select(0, a2);  // relu activation

            // calculate error
            Eigen::VectorXd error = -a2;
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
                error(it.row()) += it.value();

            err(iter) += error.array().abs().sum() / m;  // mean absolute error

            // BACKPROPAGATE
            Eigen::ArrayXd a1_prime = (a1.array() > 0).select(1, a1);  // derivative of relu
            Eigen::ArrayXd a2_prime = (a2.array() > 0).select(1, a2);  // derivative of relu
            Eigen::ArrayXd a2_delta = error.array() * a2_prime;        // length m
            Eigen::ArrayXd a1_delta = Eigen::VectorXd::Zero(k);
            for (size_t j = 0; j < m; ++j)
                a1_delta += w2.col(j).array() * a2_delta(j);
            a1_delta *= a1_prime;

            // update weights
            a1_delta *= learning_rate;
            a2_delta *= learning_rate;

            // update weights with encoder error
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it)
                for (size_t p = 0; p < k; ++p)
                    w1(p, it.row()) += a1_delta(p) * it.value();

            // update weights with decoder error
            for (size_t p = 0; p < k; ++p)
                for (size_t q = 0; q < m; ++q)
                    w2(p, q) += a1(p) * a2_delta(q);

            // apply non-negativity constraints
            w1 = (w1.array() < 0).select(0, w1);
            w2 = (w2.array() < 0).select(0, w2);
        }
        err(iter) /= n;
        if (iter > 0) curr_tol = std::abs(err(iter) - err(iter - 1)) / (err(iter) + err(iter - 1));
        if (verbose) Rprintf("%4d | %8f | %8f\n", iter + 1, err(iter), curr_tol);
        Rcpp::checkUserInterrupt();
    }
    Eigen::MatrixXd h1 = w2 * A;
    Eigen::MatrixXd h2 = w1 * A;
    w1 = w1.transpose();
    w2 = w2.transpose();
    Eigen::VectorXd d = Eigen::VectorXd::Zero(k);
    for (size_t j = 0; j < k; ++j) {
        double h1_diag = h1.row(j).array().sum();
        double w1_diag = w1.col(j).array().sum();
        d(j) = h1_diag + w1_diag;
        h1.row(j).array() /= h1_diag;
        w1.col(j).array() /= w1_diag;
        double h2_diag = h2.row(j).array().sum();
        double w2_diag = w2.col(j).array().sum();
        h2.row(j).array() /= h2_diag;
        w2.col(j).array() /= w2_diag;
    }
    err.conservativeResize(iter);
    return Rcpp::List::create(Rcpp::Named("w1") = w1, Rcpp::Named("d") = d, Rcpp::Named("h1") = h1, Rcpp::Named("w2") = w2, Rcpp::Named("h2") = h2, Rcpp::Named("error") = err);
}

// NMF FUNCTION
//[[Rcpp::export]]
Rcpp::List c_nmf(const Eigen::SparseMatrix<double> A,
                 const size_t k = 10,
                 const double tol = 1e-4,
                 const uint16_t maxit = 100,
                 const bool verbose = true,
                 const double L1 = 0,
                 const double L2 = 0,
                 const bool use_gd = false) {
    const Eigen::SparseMatrix<double> At = A.transpose();
    if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
    Eigen::MatrixXd w = Eigen::MatrixXd::Random(k, A.rows());
    Eigen::MatrixXd h = Eigen::MatrixXd::Random(k, A.cols());
    Eigen::VectorXd d(w.rows());
    double tol_ = 1;
    for (size_t iter_ = 0; iter_ < maxit && tol_ > tol; ++iter_) {
        Eigen::MatrixXd w_it = w;

        // update h, scale h, update w, scale w
        if (use_gd) {
            predict_gd(A, w, h, L1, L2);
            scale(h, d);
            predict_gd(At, h, w, L1, L2);
            scale(w, d);
        } else {
            predict_cd(A, w, h, L1, L2);
            scale(h, d);
            predict_cd(At, h, w, L1, L2);
            scale(w, d);
        }

        // calculate tolerance of the model fit to detect convergence
        tol_ = cor(w, w_it);  // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        Rcpp::checkUserInterrupt();
    }

    return Rcpp::List::create(Rcpp::Named("w") = w.transpose(), Rcpp::Named("d") = d, Rcpp::Named("h") = h);
}

// NMF FUNCTION
//[[Rcpp::export]]
size_t c_nmf_rand(const uint32_t seed, const uint32_t nrow, const uint32_t ncol, const uint32_t k, const uint16_t maxit) {
    // use inv_density = 20 to generate a 95% sparse matrix
    Eigen::SparseMatrix<double> A = rand_spmat(nrow, ncol, 20, seed);

    // time this for n different seeds
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    c_nmf(A, k, 1e-20, maxit, false, 0, 0, false);
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
    std::vector<uint32_t> nrows = {1000, 10000};
    std::vector<uint32_t> ncols = {1000, 10000, 25000};
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
