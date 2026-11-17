#pragma once

#include "fvm/core/types.hpp"

#include <vector>

namespace fvm::linalg {

/// Small row-major dense matrix for local reconstruction stencils and dense systems.
class DenseMatrix {
public:
    DenseMatrix() = default;
    DenseMatrix(Index rows, Index cols, Real init_val = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

    Index rows() const { return rows_; }
    Index cols() const { return cols_; }

    Real& operator()(Index r, Index c) { return data_[r * cols_ + c]; }
    Real operator()(Index r, Index c) const { return data_[r * cols_ + c]; }

    Real* data() { return data_.data(); }
    const Real* data() const { return data_.data(); }

    static DenseMatrix identity(Index n);
    DenseMatrix transpose() const;
    std::vector<Real> matvec(const std::vector<Real>& x) const;
    DenseMatrix matmul(const DenseMatrix& B) const;

private:
    Index rows_ = 0;
    Index cols_ = 0;
    std::vector<Real> data_;
};

/// Precomputed Householder QR decomposition of an m x n matrix (m >= n)
/// for repeated O(m*n) least-squares solves on fixed reconstruction stencils.
class QRDecomposition {
public:
    QRDecomposition() = default;

    static QRDecomposition compute(DenseMatrix A);
    static bool try_compute(DenseMatrix A, QRDecomposition& out);

    /// Solves min_x ||A x - b||_2.
    std::vector<Real> solve(std::vector<Real> b) const;

    Index rows() const { return m_; }
    Index cols() const { return n_; }
    Real min_abs_r_diag() const;
    Real max_abs_r_diag() const;

private:
    Index m_ = 0;
    Index n_ = 0;
    DenseMatrix qr_;
    std::vector<Real> r_diag_;
    std::vector<Real> betas_;
};

/// Dense vector utilities.
Real dot(const std::vector<Real>& a, const std::vector<Real>& b);
Real norm2(const std::vector<Real>& a);

/// Gaussian elimination with partial pivoting for square n x n systems Ax = b.
std::vector<Real> solve_lu(DenseMatrix A, std::vector<Real> b);

/// Householder QR least-squares solve for m x n (m >= n) systems min ||Ax - b||_2.
std::vector<Real> solve_qr_ls(DenseMatrix A, std::vector<Real> b);

}  // namespace fvm::linalg
