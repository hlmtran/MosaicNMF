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
inline float rand_unif(const uint32_t state, const uint32_t i, const uint32_t j) {
    float x = (float)rand_uint32(state, i, j) / UINT32_MAX;
    return x - std::floor(x);
}

// generate a sparse matrix
template <typename T>
Eigen::SparseMatrix<T> rand_spmat(const uint32_t nrow, const uint32_t ncol, const uint32_t inv_density, const uint32_t seed) {
    Eigen::SparseMatrix<T> mat(nrow, ncol);
    mat.reserve(Eigen::VectorXi::Constant(ncol, nrow / (inv_density - 1)));
    for (uint32_t j = 0; j < ncol; ++j) {
        for (uint32_t i = 0; i < nrow; ++i) {
            if (rand_uint32(seed, i, j) % inv_density == 0) {
                mat.insert(i, j) = (T)rand_unif(seed, i, j);
            }
        }
    }
    mat.makeCompressed();
    return mat;
}

// generate a dense matrix
template <typename T>
Eigen::Matrix<T, -1, -1> rand_mat(const uint32_t nrow, const uint32_t ncol, const uint32_t seed) {
    Eigen::Matrix<T, -1, -1> mat(nrow, ncol);
    for (uint32_t j = 0; j < ncol; ++j) {
        for (uint32_t i = 0; i < nrow; ++i) {
            mat(i, j) = (T)rand_unif(seed, i, j);
        }
    }
    return mat;
}

// NMF HELPER FUNCTIONS
// Pearson correlation between two matrices (used for determining convergence)
template <typename T>
inline T cor(Eigen::Matrix<T, -1, -1>& x, Eigen::Matrix<T, -1, -1>& y) {
    T x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
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
Eigen::MatrixXf AAt(const Eigen::MatrixXf& A) {
    Eigen::MatrixXf AAt = Eigen::MatrixXf::Zero(A.rows(), A.rows());
    AAt.selfadjointView<Eigen::Lower>().rankUpdate(A);
    AAt.triangularView<Eigen::Upper>() = AAt.transpose();
    AAt.diagonal().array() += 1e-15;  // for numerical stability during coordinate descent NNLS
    return AAt;
}

// scale rows in w (or h) to sum to 1 and put previous rowsums in d
template <typename T>
void scale(Eigen::Matrix<T, -1, -1>& w, Eigen::Matrix<T, -1, 1>& d) {
    d = w.rowwise().sum();
    d.array() += 1e-15;
    for (size_t i = 0; i < w.rows(); ++i)
        for (size_t j = 0; j < w.cols(); ++j)
            w(i, j) /= d(i);
};

// NNLS SOLVER FOR SYSTEMS IN THE FORM OF ax=b
template <typename T>
inline void nnls(Eigen::Matrix<T, -1, -1>& a, Eigen::Matrix<T, -1, 1>& b, Eigen::Matrix<T, -1, -1>& h, const size_t col, const T& L2) {
    T tol = 1;
    for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8; ++it) {
        tol = 0;
        for (size_t i = 0; i < h.rows(); ++i) {
            if (-b(i) > h(i, col)) {
                if (h(i, col) != 0) {
                    b -= a.col(i) * -h(i, col);
                    tol = 1;
                    h(i, col) = 0;
                }
            } else if (b(i) != 0) {
                h(i, col) += b(i);
                tol += std::abs(b(i) / (h(i, col) + 1e-15));
                b -= a.col(i) * b(i);
            }
        }
    }
}

// NMF UPDATE FUNCTIONS
// update h given A and w
template <typename T>
void predict(const Eigen::SparseMatrix<T>& A, const Eigen::Matrix<T, -1, -1>& w, Eigen::Matrix<T, -1, -1>& h, const T L1, const T L2) {
    Eigen::Matrix<T, -1, -1> gram_matrix = AAt(w);
    Eigen::Matrix<T, -1, 1> gram_norm = gram_matrix.diagonal().array().inverse();
    for (size_t i = 0; i < gram_norm.size(); ++i)
        gram_matrix.row(i) *= gram_norm(i);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t col = 0; col < h.cols(); ++col) {
        Eigen::Matrix<T, -1, 1> b = Eigen::Matrix<T, -1, 1>::Zero(h.rows());
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, col); it; ++it)
            b += it.value() * w.col(it.row());
        b.array() -= L1;
        b.array() *= gram_norm.array();
        nnls(gram_matrix, b, h, col, L2);
    }
}

// NMF FUNCTION
template <typename T>
Rcpp::List base_nmf(const Eigen::SparseMatrix<T>& A,
                    const size_t k = 10,
                    const T tol = 1e-4,
                    const uint16_t maxit = 100,
                    const bool verbose = true,
                    const T L1 = 0,
                    const T L2 = 0) {
    const Eigen::SparseMatrix<T> At = A.transpose();
    if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
    Eigen::Matrix<T, -1, -1> w = Eigen::Matrix<T, -1, -1>::Random(k, A.rows());
    Eigen::Matrix<T, -1, -1> h = Eigen::Matrix<T, -1, -1>::Random(k, A.cols());
    Eigen::Matrix<T, -1, 1> d(w.rows());
    T tol_ = 1;
    for (size_t iter_ = 0; iter_ < maxit && tol_ > tol; ++iter_) {
        Eigen::Matrix<T, -1, -1> w_it = w;

        // update h, scale h, update w, scale w
        predict(A, w, h, L1, L2);
        scale(h, d);
        predict(At, h, w, L1, L2);
        scale(w, d);

        // calculate tolerance of the model fit to detect convergence
        tol_ = cor(w, w_it);  // correlation between "w" across consecutive iterations
        if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
        Rcpp::checkUserInterrupt();
    }

    return Rcpp::List::create(Rcpp::Named("w") = w.transpose(), Rcpp::Named("d") = d, Rcpp::Named("h") = h);
}

//[[Rcpp::export]]
Rcpp::List c_nmf(const Eigen::SparseMatrix<float> A_float,
                 const Eigen::SparseMatrix<double> A_double,
                 const size_t k = 10,
                 const double tol = 1e-4,
                 const uint16_t maxit = 100,
                 const bool verbose = false,
                 const double L1 = 0,
                 const double L2 = 0,
                 const bool use_float = true) {
    if (use_float)
        return base_nmf(A_float, k, (float)tol, maxit, verbose, (float)L1, (float)L2);
    else
        return base_nmf(A_double, k, tol, maxit, verbose, L1, L2);
}

// NMF FUNCTION
//[[Rcpp::export]]
size_t c_nmf_rand(const uint32_t seed, const uint32_t nrow, const uint32_t ncol, const uint32_t k, const uint16_t maxit, const bool use_float) {
    // use inv_density = 20 to generate a 95% sparse matrix
    Eigen::SparseMatrix<float> A_float = rand_spmat<float>(nrow, ncol, 20, seed);
    Eigen::SparseMatrix<double> A_double = rand_spmat<double>(nrow, ncol, 20, seed);
    srand(seed);

    // time this for n different seeds
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    c_nmf(A_float, A_double, k, 1e-20, maxit, false, 0, 0, use_float);
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    size_t res = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    return res;
}

//' Automatic benchmarking using random matrices
//'
//' @export
//[[Rcpp::export]]
Rcpp::List run_benchmarking() {
    std::vector<uint32_t> seeds = {111, 222, 333, 444};
    std::vector<uint32_t> ranks = {10, 25, 50, 75, 100};
    std::vector<uint32_t> nrows = {312, 625, 1250, 2500};
    std::vector<uint32_t> ncols = {312, 625, 1250, 2500};
    std::vector<bool> use_float = {false, true};
    std::vector<uint32_t> seeds_, ranks_, nrows_, ncols_;
    std::vector<bool> use_float_;
    std::vector<size_t> times;
    for (auto seed : seeds) {
        for (auto rank : ranks) {
            for (auto nrow : nrows) {
                for (auto ncol : ncols) {
                    for (auto using_float : use_float) {
                        Rcpp::checkUserInterrupt();
                        Rprintf("seed: %8i; rank: %2i, rows: %8i, cols: %8i\n", seed, rank, nrow, ncol);
                        size_t time = c_nmf_rand(seed, nrow, ncol, rank, 100, using_float);
                        times.push_back(time);
                        seeds_.push_back(seed);
                        ranks_.push_back(rank);
                        nrows_.push_back(nrow);
                        ncols_.push_back(ncol);
                        use_float_.push_back(using_float);
                    }
                }
            }
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("times") = times,
        Rcpp::Named("seeds") = seeds_,
        Rcpp::Named("ranks") = ranks_,
        Rcpp::Named("rows") = nrows_,
        Rcpp::Named("cols") = ncols_,
        Rcpp::Named("float") = use_float_);
}

/*** R
df <- as.data.frame(run_benchmarking())
df$times <- df$times / 1e9
library(ggplot2)
ggplot(df, aes(ranks, times, color = new_algo)) +
  facet_grid(rows = vars(rows), cols = vars(cols)) +
  theme_classic() +
  geom_point() +
  stat_smooth(se = F, method = "lm") +
  scale_y_continuous(expand = c(0, 0)) +
  theme(aspect.ratio = 1) +
  labs(y = "time (sec)", x = "factorization rank", color = "with 'missing term'")
*/