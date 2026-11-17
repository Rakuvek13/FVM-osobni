#pragma once

#include "fvm/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fvm {

struct ErrorNorms {
    Real l1 = 0.0;
    Real l2 = 0.0;
    Real linf = 0.0;
};

struct FieldStats {
    Real min_val = 0.0;
    Real max_val = 0.0;
    Real total_mass = 0.0;
};

/// 2D structured cell-centered field with configurable ghost layers.
class StructuredField2D {
public:
    StructuredField2D() = default;

    StructuredField2D(Index nx, Index ny, Index nghost, Real init_val = 0.0)
        : nx_(nx),
          ny_(ny),
          nghost_(nghost),
          stride_(nx + 2 * nghost),
          data_((nx + 2 * nghost) * (ny + 2 * nghost), init_val) {}

    Index nx() const { return nx_; }
    Index ny() const { return ny_; }
    Index nghost() const { return nghost_; }
    Index stride() const { return stride_; }
    Index raw_size() const { return data_.size(); }

    Real& operator()(SignedIndex i, SignedIndex j) {
        return data_[static_cast<Index>(j + static_cast<SignedIndex>(nghost_)) * stride_ +
                     static_cast<Index>(i + static_cast<SignedIndex>(nghost_))];
    }

    Real operator()(SignedIndex i, SignedIndex j) const {
        return data_[static_cast<Index>(j + static_cast<SignedIndex>(nghost_)) * stride_ +
                     static_cast<Index>(i + static_cast<SignedIndex>(nghost_))];
    }

    Real& at(SignedIndex i, SignedIndex j) {
        check_bounds(i, j);
        return (*this)(i, j);
    }

    Real at(SignedIndex i, SignedIndex j) const {
        check_bounds(i, j);
        return (*this)(i, j);
    }

    Real* data() { return data_.data(); }
    const Real* data() const { return data_.data(); }
    std::vector<Real>& raw() { return data_; }
    const std::vector<Real>& raw() const { return data_; }

    void fill(Real val) { std::fill(data_.begin(), data_.end(), val); }

    /// Interior cell-area-weighted stats.
    FieldStats interior_stats(Real cell_area) const {
        FieldStats stats;
        if (nx_ == 0 || ny_ == 0) {
            return stats;
        }
        stats.min_val = std::numeric_limits<Real>::infinity();
        stats.max_val = -std::numeric_limits<Real>::infinity();
        Real sum = 0.0;
        for (Index j = 0; j < ny_; ++j) {
            for (Index i = 0; i < nx_; ++i) {
                const Real v = (*this)(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j));
                stats.min_val = std::min(stats.min_val, v);
                stats.max_val = std::max(stats.max_val, v);
                sum += v;
            }
        }
        stats.total_mass = sum * cell_area;
        return stats;
    }

private:
    void check_bounds(SignedIndex i, SignedIndex j) const {
        const SignedIndex g = static_cast<SignedIndex>(nghost_);
        if (i < -g || i >= static_cast<SignedIndex>(nx_) + g ||
            j < -g || j >= static_cast<SignedIndex>(ny_) + g) {
            throw std::out_of_range("StructuredField2D index out of ghost bounds");
        }
    }

    Index nx_ = 0;
    Index ny_ = 0;
    Index nghost_ = 0;
    Index stride_ = 0;
    std::vector<Real> data_;
};

inline ErrorNorms compute_error_norms(const StructuredField2D& num,
                                      const StructuredField2D& ref,
                                      Real cell_area) {
    if (num.nx() != ref.nx() || num.ny() != ref.ny()) {
        throw std::invalid_argument("Field dimension mismatch in compute_error_norms");
    }
    ErrorNorms e;
    Real sum_l1 = 0.0;
    Real sum_l2 = 0.0;
    for (Index j = 0; j < num.ny(); ++j) {
        for (Index i = 0; i < num.nx(); ++i) {
            const Real diff = std::abs(
                num(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) -
                ref(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)));
            sum_l1 += diff;
            sum_l2 += diff * diff;
            e.linf = std::max(e.linf, diff);
        }
    }
    e.l1 = sum_l1 * cell_area;
    e.l2 = std::sqrt(sum_l2 * cell_area);
    return e;
}

inline ErrorNorms compute_error_norms(const std::vector<Real>& num,
                                      const std::vector<Real>& ref,
                                      const std::vector<Real>& cell_areas) {
    if (num.size() != ref.size() || num.size() != cell_areas.size()) {
        throw std::invalid_argument("Vector size mismatch in compute_error_norms");
    }
    ErrorNorms e;
    Real sum_l1 = 0.0;
    Real sum_l2 = 0.0;
    for (Index i = 0; i < num.size(); ++i) {
        const Real diff = std::abs(num[i] - ref[i]);
        sum_l1 += cell_areas[i] * diff;
        sum_l2 += cell_areas[i] * diff * diff;
        e.linf = std::max(e.linf, diff);
    }
    e.l1 = sum_l1;
    e.l2 = std::sqrt(sum_l2);
    return e;
}

inline FieldStats compute_field_stats(const std::vector<Real>& u,
                                      const std::vector<Real>& cell_areas) {
    if (u.size() != cell_areas.size()) {
        throw std::invalid_argument("Vector size mismatch in compute_field_stats");
    }
    FieldStats stats;
    if (u.empty()) {
        return stats;
    }
    stats.min_val = std::numeric_limits<Real>::infinity();
    stats.max_val = -std::numeric_limits<Real>::infinity();
    for (Index i = 0; i < u.size(); ++i) {
        stats.min_val = std::min(stats.min_val, u[i]);
        stats.max_val = std::max(stats.max_val, u[i]);
        stats.total_mass += u[i] * cell_areas[i];
    }
    return stats;
}

}  // namespace fvm
