#include "fvm/linalg/sparse.hpp"
#include "fvm/linalg/dense.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fvm::linalg {

TripletBuilder::TripletBuilder(Index rows, Index cols, Index reserve_nnz)
    : rows_(rows), cols_(cols) {
    if (reserve_nnz > 0) {
        triplets_.reserve(reserve_nnz);
    }
}

void TripletBuilder::add(Index row, Index col, Real val) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("TripletBuilder::add index out of range");
    }
    if (val != 0.0) {
        triplets_.push_back({row, col, val});
    }
}

CSRMatrix TripletBuilder::to_csr() const {
    CSRMatrix A;
    A.rows_ = rows_;
    A.cols_ = cols_;
    A.row_ptr_.assign(rows_ + 1, 0);
    A.diag_offset_.assign(rows_, -1);

    std::vector<Triplet> sorted = triplets_;
    std::sort(sorted.begin(), sorted.end(), [](const Triplet& a, const Triplet& b) {
        return (a.r < b.r) || (a.r == b.r && a.c < b.c);
    });

    // Coalesce duplicate (r, c) pairs
    std::vector<Triplet> merged;
    merged.reserve(sorted.size());
    for (const auto& t : sorted) {
        if (!merged.empty() && merged.back().r == t.r && merged.back().c == t.c) {
            merged.back().v += t.v;
        } else {
            merged.push_back(t);
        }
    }

    A.col_idx_.reserve(merged.size());
    A.values_.reserve(merged.size());
    for (const auto& t : merged) {
        A.row_ptr_[t.r + 1] += 1;
        A.col_idx_.push_back(t.c);
        A.values_.push_back(t.v);
    }
    for (Index r = 0; r < rows_; ++r) {
        A.row_ptr_[r + 1] += A.row_ptr_[r];
        for (Index k = A.row_ptr_[r]; k < A.row_ptr_[r + 1]; ++k) {
            if (A.col_idx_[k] == r) {
                A.diag_offset_[r] = static_cast<SignedIndex>(k);
                break;
            }
        }
    }
    return A;
}

void CSRMatrix::spmv(const std::vector<Real>& x, std::vector<Real>& y) const {
    if (x.size() != cols_) {
        throw std::invalid_argument("CSRMatrix::spmv input size mismatch");
    }
    y.resize(rows_);
    for (Index i = 0; i < rows_; ++i) {
        Real sum = 0.0;
        for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            sum += values_[k] * x[col_idx_[k]];
        }
        y[i] = sum;
    }
}

std::vector<Real> CSRMatrix::operator*(const std::vector<Real>& x) const {
    std::vector<Real> y;
    spmv(x, y);
    return y;
}

void CSRMatrix::residual(const std::vector<Real>& x,
                         const std::vector<Real>& b,
                         std::vector<Real>& r) const {
    if (b.size() != rows_ || x.size() != cols_) {
        throw std::invalid_argument("CSRMatrix::residual size mismatch");
    }
    r.resize(rows_);
    for (Index i = 0; i < rows_; ++i) {
        Real ax = 0.0;
        for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            ax += values_[k] * x[col_idx_[k]];
        }
        r[i] = b[i] - ax;
    }
}

std::vector<Real> CSRMatrix::diagonal() const {
    std::vector<Real> d(rows_, 0.0);
    for (Index i = 0; i < rows_; ++i) {
        if (diag_offset_[i] >= 0) {
            d[i] = values_[static_cast<Index>(diag_offset_[i])];
        }
    }
    return d;
}

void CSRMatrix::apply_preconditioner(Preconditioner type,
                                     const std::vector<Real>& diag,
                                     const std::vector<Real>& r,
                                     std::vector<Real>& z) const {
    z.resize(rows_);
    if (type == Preconditioner::None) {
        z = r;
        return;
    }
    if (type == Preconditioner::Jacobi) {
        for (Index i = 0; i < rows_; ++i) {
            z[i] = (std::abs(diag[i]) > 1.0e-30) ? (r[i] / diag[i]) : r[i];
        }
        return;
    }

    // Symmetric Gauss-Seidel: M = (D + L) D^{-1} (D + U)
    // 1. Forward solve: (D + L) y = r
    for (Index i = 0; i < rows_; ++i) {
        Real sum = r[i];
        for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            const Index j = col_idx_[k];
            if (j >= i) {
                break;  // sorted columns
            }
            sum -= values_[k] * z[j];
        }
        const Real di = (std::abs(diag[i]) > 1.0e-30) ? diag[i] : 1.0;
        z[i] = sum / di;
    }
    // 2. Diagonal scale: y <- D y
    for (Index i = 0; i < rows_; ++i) {
        const Real di = (std::abs(diag[i]) > 1.0e-30) ? diag[i] : 1.0;
        z[i] *= di;
    }
    // 3. Backward solve: (D + U) z = y
    for (Index ii = 0; ii < rows_; ++ii) {
        const Index i = rows_ - 1 - ii;
        Real sum = z[i];
        for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            const Index j = col_idx_[k];
            if (j > i) {
                sum -= values_[k] * z[j];
            }
        }
        const Real di = (std::abs(diag[i]) > 1.0e-30) ? diag[i] : 1.0;
        z[i] = sum / di;
    }
}

LinearSolverResult solve_cg(const CSRMatrix& A,
                            const std::vector<Real>& b,
                            std::vector<Real>& x,
                            const LinearSolverConfig& cfg) {
    const Index n = A.rows();
    if (n != A.cols() || b.size() != n) {
        throw std::invalid_argument("solve_cg: matrix must be square and match rhs");
    }
    if (x.size() != n) {
        x.assign(n, 0.0);
    }

    LinearSolverResult res;
    std::vector<Real> r(n), z(n), p(n), Ap(n);
    A.residual(x, b, r);
    res.initial_residual = norm2(r);
    res.final_residual = res.initial_residual;

    const Real tol = std::max(cfg.atol, cfg.rtol * res.initial_residual);
    if (res.initial_residual <= tol) {
        res.converged = true;
        return res;
    }

    const std::vector<Real> diag = A.diagonal();
    A.apply_preconditioner(cfg.precond, diag, r, z);
    p = z;
    Real rz = dot(r, z);

    for (Index iter = 1; iter <= cfg.max_iter; ++iter) {
        A.spmv(p, Ap);
        const Real pAp = dot(p, Ap);
        if (std::abs(pAp) <= 1.0e-30) {
            res.iterations = iter - 1;
            break;
        }
        const Real alpha = rz / pAp;
        for (Index i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        res.final_residual = norm2(r);
        res.iterations = iter;
        if (res.final_residual <= tol) {
            res.converged = true;
            return res;
        }

        A.apply_preconditioner(cfg.precond, diag, r, z);
        const Real rz_new = dot(r, z);
        if (std::abs(rz) <= 1.0e-30) {
            break;
        }
        const Real beta = rz_new / rz;
        for (Index i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    return res;
}

LinearSolverResult solve_bicgstab(const CSRMatrix& A,
                                  const std::vector<Real>& b,
                                  std::vector<Real>& x,
                                  const LinearSolverConfig& cfg) {
    const Index n = A.rows();
    if (n != A.cols() || b.size() != n) {
        throw std::invalid_argument("solve_bicgstab: matrix must be square and match rhs");
    }
    if (x.size() != n) {
        x.assign(n, 0.0);
    }

    LinearSolverResult res;
    std::vector<Real> r(n), r_hat(n), p(n, 0.0), v(n, 0.0), s(n), t(n), p_hat(n), s_hat(n);
    A.residual(x, b, r);
    res.initial_residual = norm2(r);
    res.final_residual = res.initial_residual;

    const Real tol = std::max(cfg.atol, cfg.rtol * res.initial_residual);
    if (res.initial_residual <= tol) {
        res.converged = true;
        return res;
    }

    const std::vector<Real> diag = A.diagonal();
    r_hat = r;
    Real rho_prev = 1.0;
    Real alpha = 1.0;
    Real omega = 1.0;

    for (Index iter = 1; iter <= cfg.max_iter; ++iter) {
        const Real rho = dot(r_hat, r);
        if (std::abs(rho) <= 1.0e-30) {
            res.iterations = iter - 1;
            break;
        }

        if (iter == 1) {
            p = r;
        } else {
            if (std::abs(omega) <= 1.0e-30) {
                res.iterations = iter - 1;
                break;
            }
            const Real beta = (rho / rho_prev) * (alpha / omega);
            for (Index i = 0; i < n; ++i) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        A.apply_preconditioner(cfg.precond, diag, p, p_hat);
        A.spmv(p_hat, v);

        const Real r_hat_v = dot(r_hat, v);
        if (std::abs(r_hat_v) <= 1.0e-30) {
            res.iterations = iter - 1;
            break;
        }
        alpha = rho / r_hat_v;

        for (Index i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }
        const Real s_norm = norm2(s);
        if (s_norm <= tol) {
            for (Index i = 0; i < n; ++i) {
                x[i] += alpha * p_hat[i];
            }
            res.final_residual = s_norm;
            res.iterations = iter;
            res.converged = true;
            return res;
        }

        A.apply_preconditioner(cfg.precond, diag, s, s_hat);
        A.spmv(s_hat, t);

        const Real tt = dot(t, t);
        if (std::abs(tt) <= 1.0e-30) {
            res.iterations = iter - 1;
            break;
        }
        omega = dot(t, s) / tt;

        for (Index i = 0; i < n; ++i) {
            x[i] += alpha * p_hat[i] + omega * s_hat[i];
            r[i] = s[i] - omega * t[i];
        }

        res.final_residual = norm2(r);
        res.iterations = iter;
        if (res.final_residual <= tol) {
            res.converged = true;
            return res;
        }
        if (std::abs(omega) <= 1.0e-30) {
            break;
        }
        rho_prev = rho;
    }
    return res;
}

}  // namespace fvm::linalg
