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
