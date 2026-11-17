#include "fvm/linalg/dense.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fvm::linalg {

DenseMatrix DenseMatrix::identity(Index n) {
    DenseMatrix I(n, n, 0.0);
    for (Index i = 0; i < n; ++i) {
        I(i, i) = 1.0;
    }
    return I;
}

DenseMatrix DenseMatrix::transpose() const {
    DenseMatrix T(cols_, rows_, 0.0);
    for (Index i = 0; i < rows_; ++i) {
        for (Index j = 0; j < cols_; ++j) {
            T(j, i) = (*this)(i, j);
        }
    }
    return T;
}

std::vector<Real> DenseMatrix::matvec(const std::vector<Real>& x) const {
    if (x.size() != cols_) {
        throw std::invalid_argument("DenseMatrix::matvec dimension mismatch");
    }
    std::vector<Real> y(rows_, 0.0);
    for (Index i = 0; i < rows_; ++i) {
        Real acc = 0.0;
        for (Index j = 0; j < cols_; ++j) {
            acc += (*this)(i, j) * x[j];
        }
        y[i] = acc;
    }
    return y;
}

DenseMatrix DenseMatrix::matmul(const DenseMatrix& B) const {
    if (cols_ != B.rows_) {
        throw std::invalid_argument("DenseMatrix::matmul dimension mismatch");
    }
    DenseMatrix C(rows_, B.cols_, 0.0);
    for (Index i = 0; i < rows_; ++i) {
        for (Index k = 0; k < cols_; ++k) {
            const Real a_ik = (*this)(i, k);
            for (Index j = 0; j < B.cols_; ++j) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
    return C;
}

Real dot(const std::vector<Real>& a, const std::vector<Real>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: vector size mismatch");
    }
    Real s = 0.0;
    for (Index i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

Real norm2(const std::vector<Real>& a) {
    return std::sqrt(dot(a, a));
}

std::vector<Real> solve_lu(DenseMatrix A, std::vector<Real> b) {
    const Index n = A.rows();
    if (n != A.cols() || b.size() != n) {
        throw std::invalid_argument("solve_lu: matrix must be square and match rhs");
    }
    if (n == 0) {
        return {};
    }

    Real max_entry = 0.0;
    for (Index i = 0; i < n * n; ++i) {
        max_entry = std::max(max_entry, std::abs(A.data()[i]));
    }
    const Real pivot_tol = 1.0e-14 * std::max(Real{1.0}, max_entry);

    for (Index k = 0; k < n; ++k) {
        Index piv = k;
        Real max_val = std::abs(A(k, k));
        for (Index i = k + 1; i < n; ++i) {
            const Real v = std::abs(A(i, k));
            if (v > max_val) {
                max_val = v;
                piv = i;
            }
        }
        if (max_val <= pivot_tol) {
            throw std::runtime_error("solve_lu: singular or nearly singular matrix");
        }
        if (piv != k) {
            for (Index j = k; j < n; ++j) {
                std::swap(A(k, j), A(piv, j));
            }
            std::swap(b[k], b[piv]);
        }

        const Real inv_diag = 1.0 / A(k, k);
        for (Index i = k + 1; i < n; ++i) {
            const Real factor = A(i, k) * inv_diag;
            A(i, k) = 0.0;
            for (Index j = k + 1; j < n; ++j) {
                A(i, j) -= factor * A(k, j);
            }
            b[i] -= factor * b[k];
        }
    }

    std::vector<Real> x(n, 0.0);
    for (Index ii = 0; ii < n; ++ii) {
        const Index i = n - 1 - ii;
        Real sum = b[i];
        for (Index j = i + 1; j < n; ++j) {
            sum -= A(i, j) * x[j];
        }
        x[i] = sum / A(i, i);
    }
    return x;
}

namespace {

bool compute_qr_impl(DenseMatrix A,
                     Index& m_out,
                     Index& n_out,
                     DenseMatrix& qr_out,
                     std::vector<Real>& r_diag_out,
                     std::vector<Real>& betas_out,
                     bool throw_on_rank_def) {
    m_out = A.rows();
    n_out = A.cols();
    if (m_out < n_out) {
        if (throw_on_rank_def) {
            throw std::invalid_argument("QRDecomposition: requires rows >= cols");
        }
        return false;
    }
    if (n_out == 0) {
        qr_out = std::move(A);
        r_diag_out.clear();
        betas_out.clear();
        return true;
    }

    Real max_entry = 0.0;
    for (Index i = 0; i < m_out * n_out; ++i) {
        max_entry = std::max(max_entry, std::abs(A.data()[i]));
    }
    const Real rank_tol = 1.0e-13 * std::max(Real{1.0}, max_entry);

    r_diag_out.assign(n_out, 0.0);
    betas_out.assign(n_out, 0.0);

    for (Index k = 0; k < n_out; ++k) {
        Real norm_sq = 0.0;
        for (Index i = k; i < m_out; ++i) {
            norm_sq += A(i, k) * A(i, k);
        }
        const Real norm_x = std::sqrt(norm_sq);
        if (norm_x <= rank_tol) {
            if (throw_on_rank_def) {
                throw std::runtime_error("QRDecomposition: rank-deficient matrix");
            }
            return false;
        }

        const Real x0 = A(k, k);
        const Real alpha = (x0 >= 0.0) ? -norm_x : norm_x;
        const Real v0 = x0 - alpha;
        A(k, k) = 1.0;

        Real v_tail_sq = 0.0;
        for (Index i = k + 1; i < m_out; ++i) {
            A(i, k) /= v0;
            v_tail_sq += A(i, k) * A(i, k);
        }
        const Real beta = 2.0 / (1.0 + v_tail_sq);
        r_diag_out[k] = alpha;
        betas_out[k] = beta;

        for (Index j = k + 1; j < n_out; ++j) {
            Real v_dot_col = A(k, j);
            for (Index i = k + 1; i < m_out; ++i) {
                v_dot_col += A(i, k) * A(i, j);
            }
            const Real scale = beta * v_dot_col;
            A(k, j) -= scale;
            for (Index i = k + 1; i < m_out; ++i) {
                A(i, j) -= scale * A(i, k);
            }
        }
    }

    qr_out = std::move(A);
    return true;
}

}  // namespace

QRDecomposition QRDecomposition::compute(DenseMatrix A) {
    QRDecomposition out;
    compute_qr_impl(std::move(A), out.m_, out.n_, out.qr_, out.r_diag_, out.betas_, true);
    return out;
}

bool QRDecomposition::try_compute(DenseMatrix A, QRDecomposition& out) {
    return compute_qr_impl(std::move(A), out.m_, out.n_, out.qr_, out.r_diag_, out.betas_, false);
}

Real QRDecomposition::min_abs_r_diag() const {
    if (r_diag_.empty()) return 0.0;
    Real mval = std::abs(r_diag_[0]);
    for (Index i = 1; i < r_diag_.size(); ++i) {
        mval = std::min(mval, std::abs(r_diag_[i]));
    }
    return mval;
}

Real QRDecomposition::max_abs_r_diag() const {
    Real mval = 0.0;
    for (Real v : r_diag_) {
        mval = std::max(mval, std::abs(v));
    }
    return mval;
}

std::vector<Real> QRDecomposition::solve(std::vector<Real> b) const {
    if (b.size() != m_) {
        throw std::invalid_argument("QRDecomposition::solve rhs dimension mismatch");
    }
    if (n_ == 0) {
        return {};
    }

    // Compute Q^T b in-place using stored Householder vectors
    for (Index k = 0; k < n_; ++k) {
        Real v_dot_b = b[k];
        for (Index i = k + 1; i < m_; ++i) {
            v_dot_b += qr_(i, k) * b[i];
        }
        const Real scale = betas_[k] * v_dot_b;
        b[k] -= scale;
        for (Index i = k + 1; i < m_; ++i) {
            b[i] -= scale * qr_(i, k);
        }
    }

    // Back-substitution with R
    std::vector<Real> x(n_, 0.0);
    for (Index ii = 0; ii < n_; ++ii) {
        const Index i = n_ - 1 - ii;
        Real sum = b[i];
        for (Index j = i + 1; j < n_; ++j) {
            sum -= qr_(i, j) * x[j];
        }
        x[i] = sum / r_diag_[i];
    }
    return x;
}

std::vector<Real> solve_qr_ls(DenseMatrix A, std::vector<Real> b) {
    const QRDecomposition qr = QRDecomposition::compute(std::move(A));
    return qr.solve(std::move(b));
}

}  // namespace fvm::linalg
