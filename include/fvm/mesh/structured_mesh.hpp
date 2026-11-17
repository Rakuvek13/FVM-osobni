#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/types.hpp"

#include <functional>

namespace fvm {

enum class BoundarySide {
    Left,
    Right,
    Bottom,
    Top
};

enum class StructuredBCType {
    Periodic,
    ZeroGradient,      // Neumann du/dn = 0 (constant copy into ghost layer)
    Extrapolate,       // Quadratic polynomial extrapolation into ghost cells (smooth outflow)
    DirichletReflect,  // 2nd-order face reflection: u_ghost = 2*g(face) - u_interior
    DirichletExact,    // Evaluate g(x_ghost, y_ghost, t) directly at ghost cell centers
    DirichletCellAvg,  // Evaluate 2D cell average of g(x, y, t) over each ghost cell
    DirichletFaceAvg   // Exact 1D face-interval average over [s0, s1] along the boundary
};

struct StructuredSideBC {
    StructuredBCType type = StructuredBCType::ZeroGradient;
    std::function<Real(Real, Real, Real)> value = [](Real, Real, Real) { return 0.0; };
    std::function<Real(Real, Real, Real)> face_avg = [](Real, Real, Real) { return 0.0; };
};

struct StructuredBoundaryConfig {
    StructuredSideBC left;
    StructuredSideBC right;
    StructuredSideBC bottom;
    StructuredSideBC top;

    static StructuredBoundaryConfig all_periodic() {
        StructuredBoundaryConfig cfg;
        cfg.left.type = StructuredBCType::Periodic;
        cfg.right.type = StructuredBCType::Periodic;
        cfg.bottom.type = StructuredBCType::Periodic;
        cfg.top.type = StructuredBCType::Periodic;
        return cfg;
    }

    static StructuredBoundaryConfig all_zero_gradient() {
        return StructuredBoundaryConfig{};
    }
};

class StructuredMesh2D {
public:
    StructuredMesh2D() = default;
    StructuredMesh2D(Real xmin, Real xmax, Real ymin, Real ymax,
                     Index nx, Index ny, Index nghost = 1);

    Real xmin() const { return xmin_; }
    Real xmax() const { return xmax_; }
    Real ymin() const { return ymin_; }
    Real ymax() const { return ymax_; }
    Real lx() const { return xmax_ - xmin_; }
    Real ly() const { return ymax_ - ymin_; }

    Index nx() const { return nx_; }
    Index ny() const { return ny_; }
    Index nghost() const { return nghost_; }
    Index num_cells() const { return nx_ * ny_; }

    Real dx() const { return dx_; }
    Real dy() const { return dy_; }
    Real cell_area() const { return dx_ * dy_; }
    Real h_min() const { return dx_ < dy_ ? dx_ : dy_; }
    Real h_max() const { return dx_ > dy_ ? dx_ : dy_; }

    /// Cell center coordinates (supports ghost indices in [-nghost, n + nghost)).
    Vec2 cell_center(SignedIndex i, SignedIndex j) const {
        return {
            xmin_ + (static_cast<Real>(i) + 0.5) * dx_,
            ymin_ + (static_cast<Real>(j) + 0.5) * dy_
        };
    }

    /// Face center for vertical face i-1/2 (i in [0, nx]) at cell row j.
    Vec2 x_face_center(Index i, Index j) const {
        return {
            xmin_ + static_cast<Real>(i) * dx_,
            ymin_ + (static_cast<Real>(j) + 0.5) * dy_
        };
    }

    /// Face center for horizontal face j-1/2 (j in [0, ny]) at cell column i.
    Vec2 y_face_center(Index i, Index j) const {
        return {
            xmin_ + (static_cast<Real>(i) + 0.5) * dx_,
            ymin_ + static_cast<Real>(j) * dy_
        };
    }

    StructuredField2D create_field(Real init_val = 0.0) const {
        return StructuredField2D(nx_, ny_, nghost_, init_val);
    }

    /// Initializes cell averages using n_pts x n_pts tensor-product Gauss-Legendre quadrature.
    void initialize_cell_averages(StructuredField2D& field,
                                  const std::function<Real(Real, Real)>& f,
                                  int gauss_pts_per_dim = 4) const;

    /// Populates left/right (X) ghost layers for rows j in [0, ny).
    void apply_boundary_conditions_x(StructuredField2D& field,
                                     const StructuredBoundaryConfig& bc,
                                     Real t,
                                     bool apply_left,
                                     bool apply_right) const;

    /// Populates bottom/top (Y) ghost layers across the full padded stride i in [-g, nx + g).
    void apply_boundary_conditions_y(StructuredField2D& field,
                                     const StructuredBoundaryConfig& bc,
                                     Real t,
                                     bool apply_bottom,
                                     bool apply_top) const;

    /// Populates ghost cells according to boundary conditions at simulation time t.
    void apply_boundary_conditions(StructuredField2D& field,
                                   const StructuredBoundaryConfig& bc,
                                   Real t = 0.0) const;

private:
    Real xmin_ = 0.0;
    Real xmax_ = 1.0;
    Real ymin_ = 0.0;
    Real ymax_ = 1.0;
    Index nx_ = 0;
    Index ny_ = 0;
    Index nghost_ = 1;
    Real dx_ = 0.0;
    Real dy_ = 0.0;
};

}  // namespace fvm
