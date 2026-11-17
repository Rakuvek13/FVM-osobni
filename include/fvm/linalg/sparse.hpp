#pragma once

#include "fvm/core/types.hpp"

#include <functional>
#include <vector>

namespace fvm::linalg {

enum class Preconditioner {
    None,
    Jacobi,
    SGS  // Symmetric Gauss-Seidel: M = (D + L) D^{-1} (D + U)
};

struct LinearSolverConfig {
    Real rtol = 1.0e-10;
    Real atol = 1.0e-14;
    Index max_iter = 5000;
    Preconditioner precond = Preconditioner::SGS;
};

struct LinearSolverResult {
    bool converged = false;
    Index iterations = 0;
    Real initial_residual = 0.0;
    Real final_residual = 0.0;
};

/// Compressed Sparse Row matrix with sorted column indices per row.
class CSRMatrix {
public:
    CSRMatrix() = default;

    Index rows() const { return rows_; }
    Index cols() const { return cols_; }
    Index nnz() const { return values_.size(); }

    const std::vector<Index>& row_ptr() const { return row_ptr_; }
    const std::vector<Index>& col_idx() const { return col_idx_; }
    const std::vector<Real>& values() const { return values_; }

    /// y = A * x
    void spmv(const std::vector<Real>& x, std::vector<Real>& y) const;
    std::vector<Real> operator*(const std::vector<Real>& x) const;

    /// r = b - A * x
    void residual(const std::vector<Real>& x,
                  const std::vector<Real>& b,
                  std::vector<Real>& r) const;

    /// Extracts main diagonal entries (0.0 if missing).
    std::vector<Real> diagonal() const;

    /// Applies M^{-1} r -> z for Jacobi or SGS.
    void apply_preconditioner(Preconditioner type,
                              const std::vector<Real>& diag,
                              const std::vector<Real>& r,
                              std::vector<Real>& z) const;

private:
    friend class TripletBuilder;

    Index rows_ = 0;
    Index cols_ = 0;
    std::vector<Index> row_ptr_;
    std::vector<Index> col_idx_;
    std::vector<Real> values_;
    std::vector<SignedIndex> diag_offset_;  // -1 if diagonal absent in row
};

/// Accumulates (row, col, value) triplets and assembles a canonical CSRMatrix
/// (merging duplicates and sorting columns per row).
class TripletBuilder {
public:
    TripletBuilder(Index rows, Index cols, Index reserve_nnz = 0);

    void add(Index row, Index col, Real val);
    CSRMatrix to_csr() const;

private:
    struct Triplet {
        Index r;
        Index c;
        Real v;
    };
    Index rows_ = 0;
    Index cols_ = 0;
    std::vector<Triplet> triplets_;
};

/// Preconditioned Conjugate Gradient for symmetric positive-definite matrices.
LinearSolverResult solve_cg(const CSRMatrix& A,
                            const std::vector<Real>& b,
                            std::vector<Real>& x,
                            const LinearSolverConfig& cfg = {});

/// Preconditioned BiCGSTAB for general nonsymmetric sparse matrices.
LinearSolverResult solve_bicgstab(const CSRMatrix& A,
                                  const std::vector<Real>& b,
                                  std::vector<Real>& x,
                                  const LinearSolverConfig& cfg = {});

}  // namespace fvm::linalg
