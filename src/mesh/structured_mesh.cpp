#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/numerics/quadrature.hpp"

#include <stdexcept>

namespace fvm {

namespace {

Real cell_average_rect(const std::function<Real(Real, Real)>& f,
                       Real x0, Real dx, Real y0, Real dy,
                       const std::vector<quadrature::QuadPoint1D>& q1d) {
    Real avg = 0.0;
    for (const auto& qy : q1d) {
        const Real y = y0 + qy.xi * dy;
        for (const auto& qx : q1d) {
            const Real x = x0 + qx.xi * dx;
            avg += qx.weight * qy.weight * f(x, y);
        }
    }
    return avg;
}

Real ghost_cell_average(const std::function<Real(Real, Real, Real)>& f,
                        Real x0, Real dx, Real y0, Real dy, Real t,
                        const std::vector<quadrature::QuadPoint1D>& q1d) {
    Real avg = 0.0;
    for (const auto& qy : q1d) {
        const Real y = y0 + qy.xi * dy;
        for (const auto& qx : q1d) {
            const Real x = x0 + qx.xi * dx;
            avg += qx.weight * qy.weight * f(x, y, t);
        }
    }
    return avg;
}

inline Real cubic_extrap(Real v1, Real v2, Real v3, Real v4) {
    return 4.0 * v1 - 6.0 * v2 + 4.0 * v3 - v4;
}

}  // namespace

StructuredMesh2D::StructuredMesh2D(Real xmin, Real xmax, Real ymin, Real ymax,
                                   Index nx, Index ny, Index nghost)
    : xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      nx_(nx),
      ny_(ny),
      nghost_(nghost) {
    if (xmax_ <= xmin_ || ymax_ <= ymin_) {
        throw std::invalid_argument("StructuredMesh2D: invalid domain bounds");
    }
    if (nx_ == 0 || ny_ == 0) {
        throw std::invalid_argument("StructuredMesh2D: nx and ny must be positive");
    }
    dx_ = (xmax_ - xmin_) / static_cast<Real>(nx_);
    dy_ = (ymax_ - ymin_) / static_cast<Real>(ny_);
}

void StructuredMesh2D::initialize_cell_averages(
    StructuredField2D& field,
    const std::function<Real(Real, Real)>& f,
    int gauss_pts_per_dim) const {
    if (field.nx() != nx_ || field.ny() != ny_) {
        throw std::invalid_argument("initialize_cell_averages: field dimensions do not match mesh");
    }
    const auto q1d = quadrature::gauss_legendre_01(gauss_pts_per_dim);
    for (Index j = 0; j < ny_; ++j) {
        const Real y0 = ymin_ + static_cast<Real>(j) * dy_;
        for (Index i = 0; i < nx_; ++i) {
            const Real x0 = xmin_ + static_cast<Real>(i) * dx_;
            field(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) =
                cell_average_rect(f, x0, dx_, y0, dy_, q1d);
        }
    }
}

void StructuredMesh2D::apply_boundary_conditions_x(
    StructuredField2D& field,
    const StructuredBoundaryConfig& bc,
    Real t,
    bool apply_left,
    bool apply_right) const {
    if (field.nx() != nx_ || field.ny() != ny_) {
        throw std::invalid_argument("apply_boundary_conditions_x: field dimensions do not match mesh");
    }
    const SignedIndex g = static_cast<SignedIndex>(field.nghost());
    const SignedIndex nx_s = static_cast<SignedIndex>(nx_);
    const SignedIndex ny_s = static_cast<SignedIndex>(ny_);

    if (g == 0 || (!apply_left && !apply_right)) {
        return;
    }

    const bool need_q4 =
        (apply_left && bc.left.type == StructuredBCType::DirichletCellAvg) ||
        (apply_right && bc.right.type == StructuredBCType::DirichletCellAvg);
    const auto q4 = need_q4
        ? quadrature::gauss_legendre_01(4)
        : std::vector<quadrature::QuadPoint1D>{};

    // X-direction ghost cells (Left & Right)
    for (SignedIndex j = 0; j < ny_s; ++j) {
        const Real y0 = ymin_ + static_cast<Real>(j) * dy_;
        const Real yj = y0 + 0.5 * dy_;
        for (SignedIndex k = 0; k < g; ++k) {
            if (apply_left) {
                const SignedIndex il = -1 - k;
                switch (bc.left.type) {
                    case StructuredBCType::Periodic:
                        field(il, j) = field((nx_s - 1 - k % nx_s + nx_s) % nx_s, j);
                        break;
                    case StructuredBCType::ZeroGradient:
                        field(il, j) = field(0, j);
                        break;
                    case StructuredBCType::Extrapolate:
                        field(il, j) = (nx_s >= 4)
                            ? cubic_extrap(field(il + 1, j), field(il + 2, j), field(il + 3, j), field(il + 4, j))
                            : field(0, j);
                        break;
                    case StructuredBCType::DirichletReflect: {
                        const SignedIndex src = std::min(k, nx_s - 1);
                        const Real bv = bc.left.value(xmin_, yj, t);
                        field(il, j) = 2.0 * bv - field(src, j);
                        break;
                    }
                    case StructuredBCType::DirichletExact: {
                        const Vec2 xc = cell_center(il, j);
                        field(il, j) = bc.left.value(xc.x, xc.y, t);
                        break;
                    }
                    case StructuredBCType::DirichletCellAvg: {
                        const Real x0 = xmin_ + static_cast<Real>(il) * dx_;
                        field(il, j) = ghost_cell_average(bc.left.value, x0, dx_, y0, dy_, t, q4);
                        break;
                    }
                    case StructuredBCType::DirichletFaceAvg: {
                        const Real y1 = y0 + dy_;
                        field(il, j) = bc.left.face_avg(y0, y1, t);
                        break;
                    }
                }
            }

            if (apply_right) {
                const SignedIndex ir = nx_s + k;
                switch (bc.right.type) {
                    case StructuredBCType::Periodic:
                        field(ir, j) = field(k % nx_s, j);
                        break;
                    case StructuredBCType::ZeroGradient:
                        field(ir, j) = field(nx_s - 1, j);
                        break;
                    case StructuredBCType::Extrapolate:
                        field(ir, j) = (nx_s >= 4)
                            ? cubic_extrap(field(ir - 1, j), field(ir - 2, j), field(ir - 3, j), field(ir - 4, j))
                            : field(nx_s - 1, j);
                        break;
                    case StructuredBCType::DirichletReflect: {
                        const SignedIndex src = std::max(SignedIndex{0}, nx_s - 1 - k);
                        const Real bv = bc.right.value(xmax_, yj, t);
                        field(ir, j) = 2.0 * bv - field(src, j);
                        break;
                    }
                    case StructuredBCType::DirichletExact: {
                        const Vec2 xc = cell_center(ir, j);
                        field(ir, j) = bc.right.value(xc.x, xc.y, t);
                        break;
                    }
                    case StructuredBCType::DirichletCellAvg: {
                        const Real x0 = xmin_ + static_cast<Real>(ir) * dx_;
                        field(ir, j) = ghost_cell_average(bc.right.value, x0, dx_, y0, dy_, t, q4);
                        break;
                    }
                    case StructuredBCType::DirichletFaceAvg: {
                        const Real y1 = y0 + dy_;
                        field(ir, j) = bc.right.face_avg(y0, y1, t);
                        break;
                    }
                }
            }
        }
    }
}

void StructuredMesh2D::apply_boundary_conditions_y(
    StructuredField2D& field,
    const StructuredBoundaryConfig& bc,
    Real t,
    bool apply_bottom,
    bool apply_top) const {
    if (field.nx() != nx_ || field.ny() != ny_) {
        throw std::invalid_argument("apply_boundary_conditions_y: field dimensions do not match mesh");
    }
    const SignedIndex g = static_cast<SignedIndex>(field.nghost());
    const SignedIndex nx_s = static_cast<SignedIndex>(nx_);
    const SignedIndex ny_s = static_cast<SignedIndex>(ny_);

    if (g == 0 || (!apply_bottom && !apply_top)) {
        return;
    }

    const bool need_q4 =
        (apply_bottom && bc.bottom.type == StructuredBCType::DirichletCellAvg) ||
        (apply_top && bc.top.type == StructuredBCType::DirichletCellAvg);
    const auto q4 = need_q4
        ? quadrature::gauss_legendre_01(4)
        : std::vector<quadrature::QuadPoint1D>{};

    // Y-direction ghost cells (Bottom & Top) including corner ghost cells [-g, nx_s + g)
    for (SignedIndex i = -g; i < nx_s + g; ++i) {
        const Real x0 = xmin_ + static_cast<Real>(i) * dx_;
        const Real xi = x0 + 0.5 * dx_;
        for (SignedIndex k = 0; k < g; ++k) {
            if (apply_bottom) {
                const SignedIndex jb = -1 - k;
                switch (bc.bottom.type) {
                    case StructuredBCType::Periodic:
                        field(i, jb) = field(i, (ny_s - 1 - k % ny_s + ny_s) % ny_s);
                        break;
                    case StructuredBCType::ZeroGradient:
                        field(i, jb) = field(i, 0);
                        break;
                    case StructuredBCType::Extrapolate:
                        field(i, jb) = (ny_s >= 4)
                            ? cubic_extrap(field(i, jb + 1), field(i, jb + 2), field(i, jb + 3), field(i, jb + 4))
                            : field(i, 0);
                        break;
                    case StructuredBCType::DirichletReflect: {
                        const SignedIndex src = std::min(k, ny_s - 1);
                        const Real bv = bc.bottom.value(xi, ymin_, t);
                        field(i, jb) = 2.0 * bv - field(i, src);
                        break;
                    }
                    case StructuredBCType::DirichletExact: {
                        const Vec2 xc = cell_center(i, jb);
                        field(i, jb) = bc.bottom.value(xc.x, xc.y, t);
                        break;
                    }
                    case StructuredBCType::DirichletCellAvg: {
                        const Real y0 = ymin_ + static_cast<Real>(jb) * dy_;
                        field(i, jb) = ghost_cell_average(bc.bottom.value, x0, dx_, y0, dy_, t, q4);
                        break;
                    }
                    case StructuredBCType::DirichletFaceAvg: {
                        const Real x1 = x0 + dx_;
                        field(i, jb) = bc.bottom.face_avg(x0, x1, t);
                        break;
                    }
                }
            }

            if (apply_top) {
                const SignedIndex jt = ny_s + k;
                switch (bc.top.type) {
                    case StructuredBCType::Periodic:
                        field(i, jt) = field(i, k % ny_s);
                        break;
                    case StructuredBCType::ZeroGradient:
                        field(i, jt) = field(i, ny_s - 1);
                        break;
                    case StructuredBCType::Extrapolate:
                        field(i, jt) = (ny_s >= 4)
                            ? cubic_extrap(field(i, jt - 1), field(i, jt - 2), field(i, jt - 3), field(i, jt - 4))
                            : field(i, ny_s - 1);
                        break;
                    case StructuredBCType::DirichletReflect: {
                        const SignedIndex src = std::max(SignedIndex{0}, ny_s - 1 - k);
                        const Real bv = bc.top.value(xi, ymax_, t);
                        field(i, jt) = 2.0 * bv - field(i, src);
                        break;
                    }
                    case StructuredBCType::DirichletExact: {
                        const Vec2 xc = cell_center(i, jt);
                        field(i, jt) = bc.top.value(xc.x, xc.y, t);
                        break;
                    }
                    case StructuredBCType::DirichletCellAvg: {
                        const Real y0 = ymin_ + static_cast<Real>(jt) * dy_;
                        field(i, jt) = ghost_cell_average(bc.top.value, x0, dx_, y0, dy_, t, q4);
                        break;
                    }
                    case StructuredBCType::DirichletFaceAvg: {
                        const Real x1 = x0 + dx_;
                        field(i, jt) = bc.top.face_avg(x0, x1, t);
                        break;
                    }
                }
            }
        }
    }
}

void StructuredMesh2D::apply_boundary_conditions(
    StructuredField2D& field,
    const StructuredBoundaryConfig& bc,
    Real t) const {
    apply_boundary_conditions_x(field, bc, t, true, true);
    apply_boundary_conditions_y(field, bc, t, true, true);
}

}  // namespace fvm
