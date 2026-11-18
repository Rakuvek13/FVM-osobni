#include "fvm/solvers/structured_solver.hpp"
#include "fvm/io/vtk_writer.hpp"
#include "fvm/numerics/flux.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fvm::solvers {

namespace {

inline Real minmod2(Real a, Real b) {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return (std::abs(a) < std::abs(b)) ? a : b;
}

inline Real minmod3(Real a, Real b, Real c) {
    if (a > 0.0 && b > 0.0 && c > 0.0) {
        return std::min({a, b, c});
    }
    if (a < 0.0 && b < 0.0 && c < 0.0) {
        return std::max({a, b, c});
    }
    return 0.0;
}

/// Limited cell slope sigma_i approximating dx * (du/dx)_i.
inline Real tvd_slope(Real dm, Real dp, TVDSlopeLimiter limiter) {
    if (dm * dp <= 0.0) {
        return 0.0;
    }
    switch (limiter) {
        case TVDSlopeLimiter::Minmod:
            return minmod2(dm, dp);
        case TVDSlopeLimiter::VanLeer:
            return (2.0 * dm * dp) / (dm + dp);
        case TVDSlopeLimiter::MC:
            return minmod3(0.5 * (dm + dp), 2.0 * dm, 2.0 * dp);
        case TVDSlopeLimiter::Superbee: {
            const Real s = (dm > 0.0) ? 1.0 : -1.0;
            const Real am = std::abs(dm);
            const Real ap = std::abs(dp);
            return s * std::max(std::min(2.0 * am, ap), std::min(am, 2.0 * ap));
        }
    }
    return 0.0;
}

inline void apply_muscl_tvd(Real& uL, Real& uR, Real u_ll, Real u_rr, TVDSlopeLimiter lim) {
    const Real du_c = uR - uL;
    const Real sL = tvd_slope(uL - u_ll, du_c, lim);
    const Real sR = tvd_slope(du_c, u_rr - uR, lim);
    uL += 0.5 * sL;
    uR -= 0.5 * sR;
}

inline Real eno2_slope(Real dm, Real dp) {
    return (std::abs(dm) < std::abs(dp)) ? dm : dp;
}

inline void apply_eno2(Real& uL, Real& uR, Real u_ll, Real u_rr) {
    const Real du_c = uR - uL;
    const Real sL = eno2_slope(uL - u_ll, du_c);
    const Real sR = eno2_slope(du_c, u_rr - uR);
    uL += 0.5 * sL;
    uR -= 0.5 * sR;
}

/// Shu (1997) Algorithm 2.1 adaptive ENO3 stencil shift r in {0, 1, 2} for cell index 0
/// with uniform cell averages {um2, um1, u0, u1, u2}.
inline int select_eno3_shift(Real um2, Real um1, Real u0, Real u1, Real u2) {
    const Real d1_L = u0 - um1;
    const Real d1_R = u1 - u0;
    if (std::abs(d1_L) >= std::abs(d1_R)) {
        const Real d2_0 = u1 - 2.0 * u0 + um1;
        const Real d2_1 = u2 - 2.0 * u1 + u0;
        return (std::abs(d2_0) < std::abs(d2_1)) ? 1 : 0;
    } else {
        const Real d2_m1 = u0 - 2.0 * um1 + um2;
        const Real d2_0 = u1 - 2.0 * u0 + um1;
        return (std::abs(d2_m1) < std::abs(d2_0)) ? 2 : 1;
    }
}

/// Reconstructs 3rd-order left and right interface values of cell 0 from uniform cell averages.
inline void eno3_cell_faces(Real um2, Real um1, Real u0, Real u1, Real u2,
                            Real& u_left_face, Real& u_right_face) {
    const int r = select_eno3_shift(um2, um1, u0, u1, u2);
    constexpr Real inv6 = 1.0 / 6.0;
    if (r == 0) {
        u_left_face  = (11.0 * u0 - 7.0 * u1 + 2.0 * u2) * inv6;
        u_right_face = (2.0 * u0 + 5.0 * u1 - u2) * inv6;
    } else if (r == 1) {
        u_left_face  = (2.0 * um1 + 5.0 * u0 - u1) * inv6;
        u_right_face = (-um1 + 5.0 * u0 + 2.0 * u1) * inv6;
    } else {
        u_left_face  = (-um2 + 5.0 * um1 + 2.0 * u0) * inv6;
        u_right_face = (2.0 * um2 - 7.0 * um1 + 11.0 * u0) * inv6;
    }
}

/// Evaluates quadratic ENO3 polynomial inside cell 0 at the 2-point Gauss-Legendre nodes
/// xi = +- 1 / (2 sqrt(3)), where p(+- 1/(2 sqrt(3))) = u0 +- p'(0) / (2 sqrt(3)) identically.
inline void eno3_gauss2(Real um2, Real um1, Real u0, Real u1, Real u2,
                        Real& u_gm, Real& u_gp) {
    const int r = select_eno3_shift(um2, um1, u0, u1, u2);
    Real c1 = 0.0;
    if (r == 0) {
        c1 = (-3.0 * u0 + 4.0 * u1 - u2) * 0.5;
    } else if (r == 1) {
        c1 = (u1 - um1) * 0.5;
    } else {
        c1 = (3.0 * u0 - 4.0 * um1 + um2) * 0.5;
    }
    static const Real kInv2Sqrt3 = 0.5 / std::sqrt(3.0);
    const Real shift = c1 * kInv2Sqrt3;
    u_gm = u0 - shift;
    u_gp = u0 + shift;
}

inline void weno5_smoothness(Real qm2, Real qm1, Real q0, Real qp1, Real qp2,
                             Real& b0, Real& b1, Real& b2) {
    const Real t0a = qm2 - 2.0 * qm1 + q0;
    const Real t0b = qm2 - 4.0 * qm1 + 3.0 * q0;
    b0 = (13.0 / 12.0) * t0a * t0a + 0.25 * t0b * t0b;

    const Real t1a = qm1 - 2.0 * q0 + qp1;
    const Real t1b = qm1 - qp1;
    b1 = (13.0 / 12.0) * t1a * t1a + 0.25 * t1b * t1b;

    const Real t2a = q0 - 2.0 * qp1 + qp2;
    const Real t2b = 3.0 * q0 - 4.0 * qp1 + qp2;
    b2 = (13.0 / 12.0) * t2a * t2a + 0.25 * t2b * t2b;
}

inline void weno5_unnorm_factors(Real b0, Real b1, Real b2,
                                 WENOWeightType wtype, Real eps,
                                 Real& s0, Real& s1, Real& s2) {
    if (wtype == WENOWeightType::WENO_Z) {
        const Real tau5 = std::abs(b0 - b2);
        const Real r0 = tau5 / (b0 + eps);
        const Real r1 = tau5 / (b1 + eps);
        const Real r2 = tau5 / (b2 + eps);
        s0 = 1.0 + r0 * r0;
        s1 = 1.0 + r1 * r1;
        s2 = 1.0 + r2 * r2;
    } else {
        const Real d0 = b0 + eps;
        const Real d1 = b1 + eps;
        const Real d2 = b2 + eps;
        s0 = 1.0 / (d0 * d0);
        s1 = 1.0 / (d1 * d1);
        s2 = 1.0 / (d2 * d2);
    }
}

/// Computes 5th-order WENO5 left (+, i-1/2) and right (-, i+1/2) cell interface averages.
inline void weno5_cell_faces(Real qm2, Real qm1, Real q0, Real qp1, Real qp2,
                             WENOWeightType wtype, Real eps,
                             Real& u_left_face, Real& u_right_face) {
    Real b0 = 0.0, b1 = 0.0, b2 = 0.0;
    weno5_smoothness(qm2, qm1, q0, qp1, qp2, b0, b1, b2);
    Real s0 = 0.0, s1 = 0.0, s2 = 0.0;
    weno5_unnorm_factors(b0, b1, b2, wtype, eps, s0, s1, s2);

    constexpr Real inv6 = 1.0 / 6.0;
    const Real pL0 = (-qm2 + 5.0 * qm1 + 2.0 * q0) * inv6;
    const Real pL1 = (2.0 * qm1 + 5.0 * q0 - qp1) * inv6;
    const Real pL2 = (11.0 * q0 - 7.0 * qp1 + 2.0 * qp2) * inv6;
    const Real wL0 = 0.3 * s0, wL1 = 0.6 * s1, wL2 = 0.1 * s2;
    u_left_face = (wL0 * pL0 + wL1 * pL1 + wL2 * pL2) / (wL0 + wL1 + wL2);

    const Real pR0 = (2.0 * qm2 - 7.0 * qm1 + 11.0 * q0) * inv6;
    const Real pR1 = (-qm1 + 5.0 * q0 + 2.0 * qp1) * inv6;
    const Real pR2 = (2.0 * q0 + 5.0 * qp1 - qp2) * inv6;
    const Real wR0 = 0.1 * s0, wR1 = 0.6 * s1, wR2 = 0.3 * s2;
    u_right_face = (wR0 * pR0 + wR1 * pR1 + wR2 * pR2) / (wR0 + wR1 + wR2);
}

/// Evaluates 5th-order WENO5 at the 3-pt Gauss-Legendre nodes xi = {-sqrt(15)/10, 0, +sqrt(15)/10}
/// on [-1/2, 1/2], using positive optimal weights at the outer nodes and Shi-Hu-Shu (2002)
/// negative-weight splitting at the midpoint xi = 0.
inline void weno5_gauss3(Real qm2, Real qm1, Real q0, Real qp1, Real qp2,
                         WENOWeightType wtype, Real eps,
                         Real& u_gm, Real& u_g0, Real& u_gp) {
    Real b0 = 0.0, b1 = 0.0, b2 = 0.0;
    weno5_smoothness(qm2, qm1, q0, qp1, qp2, b0, b1, b2);
    Real s0 = 0.0, s1 = 0.0, s2 = 0.0;
    weno5_unnorm_factors(b0, b1, b2, wtype, eps, s0, s1, s2);

    const Real d1_0 = 0.5 * (3.0 * q0 - 4.0 * qm1 + qm2);
    const Real d2_0 = 0.5 * (q0 - 2.0 * qm1 + qm2);
    const Real d1_1 = 0.5 * (qp1 - qm1);
    const Real d2_1 = 0.5 * (qm1 - 2.0 * q0 + qp1);
    const Real d1_2 = 0.5 * (-3.0 * q0 + 4.0 * qp1 - qp2);
    const Real d2_2 = 0.5 * (q0 - 2.0 * qp1 + qp2);

    static const Real kSqrt15 = std::sqrt(15.0);
    static const Real kXiOut = 0.1 * kSqrt15;      // sqrt(3/5) / 2
    constexpr Real kX2Shift = 1.0 / 15.0;          // kXiOut^2 - 1/12 = 3/20 - 1/12
    constexpr Real kInv12 = 1.0 / 12.0;

    static const Real dm0 = 126.0 / 655.0 + (71.0 / 5240.0) * kSqrt15;
    static const Real dm1 = 403.0 / 655.0;
    static const Real dm2 = 126.0 / 655.0 - (71.0 / 5240.0) * kSqrt15;

    const Real base0 = q0 + d2_0 * kX2Shift;
    const Real base1 = q0 + d2_1 * kX2Shift;
    const Real base2 = q0 + d2_2 * kX2Shift;
    const Real sh0 = d1_0 * kXiOut;
    const Real sh1 = d1_1 * kXiOut;
    const Real sh2 = d1_2 * kXiOut;

    const Real pm0 = base0 - sh0, pm1 = base1 - sh1, pm2 = base2 - sh2;
    const Real wm0 = dm0 * s0, wm1 = dm1 * s1, wm2 = dm2 * s2;
    u_gm = (wm0 * pm0 + wm1 * pm1 + wm2 * pm2) / (wm0 + wm1 + wm2);

    const Real pp0 = base0 + sh0, pp1 = base1 + sh1, pp2 = base2 + sh2;
    const Real wp0 = dm2 * s0, wp1 = dm1 * s1, wp2 = dm0 * s2;
    u_gp = (wp0 * pp0 + wp1 * pp1 + wp2 * pp2) / (wp0 + wp1 + wp2);

    const Real p0_0 = q0 - d2_0 * kInv12;
    const Real p0_1 = q0 - d2_1 * kInv12;
    const Real p0_2 = q0 - d2_2 * kInv12;
    constexpr Real sig_p = 214.0 / 80.0;
    constexpr Real sig_m = 134.0 / 80.0;
    const Real wp0_0 = (9.0 / 214.0) * s0, wp0_1 = (196.0 / 214.0) * s1, wp0_2 = (9.0 / 214.0) * s2;
    const Real wm0_0 = (9.0 / 67.0) * s0,  wm0_1 = (49.0 / 67.0) * s1,   wm0_2 = (9.0 / 67.0) * s2;
    const Real vp = (wp0_0 * p0_0 + wp0_1 * p0_1 + wp0_2 * p0_2) / (wp0_0 + wp0_1 + wp0_2);
    const Real vm = (wm0_0 * p0_0 + wm0_1 * p0_1 + wm0_2 * p0_2) / (wm0_0 + wm0_1 + wm0_2);
    u_g0 = sig_p * vp - sig_m * vm;
}

inline void weno3_unnorm_factors(Real b0, Real b1,
                                 WENOWeightType wtype, Real eps,
                                 Real& s0, Real& s1) {
    if (wtype == WENOWeightType::WENO_Z) {
        const Real tau3 = std::abs(b1 - b0);
        const Real r0 = tau3 / (b0 + eps);
        const Real r1 = tau3 / (b1 + eps);
        s0 = 1.0 + r0 * r0;
        s1 = 1.0 + r1 * r1;
    } else {
        const Real d0 = b0 + eps;
        const Real d1 = b1 + eps;
        s0 = 1.0 / (d0 * d0);
        s1 = 1.0 / (d1 * d1);
    }
}

/// Computes 3rd-order WENO3 left (+, i-1/2) and right (-, i+1/2) cell interface averages.
inline void weno3_cell_faces(Real qm1, Real q0, Real qp1,
                             WENOWeightType wtype, Real eps,
                             Real& u_left_face, Real& u_right_face) {
    const Real d0 = q0 - qm1;
    const Real d1 = qp1 - q0;
    const Real b0 = d0 * d0;
    const Real b1 = d1 * d1;
    Real s0 = 0.0, s1 = 0.0;
    weno3_unnorm_factors(b0, b1, wtype, eps, s0, s1);

    const Real pL0 = 0.5 * (qm1 + q0);
    const Real pL1 = 0.5 * (3.0 * q0 - qp1);
    const Real wL0 = 2.0 * s0;
    const Real wL1 = s1;
    u_left_face = (wL0 * pL0 + wL1 * pL1) / (wL0 + wL1);

    const Real pR0 = 0.5 * (3.0 * q0 - qm1);
    const Real pR1 = 0.5 * (q0 + qp1);
    const Real wR0 = s0;
    const Real wR1 = 2.0 * s1;
    u_right_face = (wR0 * pR0 + wR1 * pR1) / (wR0 + wR1);
}

/// Evaluates 3rd-order WENO3 at the 2-point Gauss-Legendre nodes xi = +-1 / (2 sqrt(3))
/// where optimal linear weights for the two 2-cell stencils are both 1/2.
inline void weno3_gauss2(Real qm1, Real q0, Real qp1,
                         WENOWeightType wtype, Real eps,
                         Real& u_gm, Real& u_gp) {
    const Real d0 = q0 - qm1;
    const Real d1 = qp1 - q0;
    const Real b0 = d0 * d0;
    const Real b1 = d1 * d1;
    Real s0 = 0.0, s1 = 0.0;
    weno3_unnorm_factors(b0, b1, wtype, eps, s0, s1);

    const Real c1 = (s0 * d0 + s1 * d1) / (s0 + s1);
    static const Real kInv2Sqrt3 = 0.5 / std::sqrt(3.0);
    const Real shift = c1 * kInv2Sqrt3;
    u_gm = q0 - shift;
    u_gp = q0 + shift;
}

}  // namespace

StructuredConservationLaw make_linear_advection_law(Real ax, Real ay) {
    StructuredConservationLaw law;
    law.fx = [ax](Real, Real, Real u) { return ax * u; };
    law.fy = [ay](Real, Real, Real u) { return ay * u; };
    const Real abs_ax = std::abs(ax);
    const Real abs_ay = std::abs(ay);
    law.alpha_x = [abs_ax](Real, Real, Real, Real) { return abs_ax; };
    law.alpha_y = [abs_ay](Real, Real, Real, Real) { return abs_ay; };
    law.max_speeds = [abs_ax, abs_ay](const StructuredField2D&, Real& max_ax, Real& max_ay) {
        max_ax = abs_ax;
        max_ay = abs_ay;
    };
    return law;
}

StructuredFVSolver::StructuredFVSolver(const StructuredMesh2D& mesh,
                                       StructuredBoundaryConfig bc,
                                       StructuredConservationLaw law,
                                       ReconstructionScheme scheme,
                                       TVDSlopeLimiter limiter,
                                       int eno_order,
                                       WENOWeightType weno_type,
                                       const mpi::CartDecomp2D* decomp,
                                       int weno_order)
    : mesh_(mesh),
      bc_(std::move(bc)),
      law_(std::move(law)),
      scheme_(scheme),
      limiter_(limiter),
      eno_order_(eno_order),
      weno_type_(weno_type),
      weno_order_(weno_order),
      decomp_(decomp) {
    if (scheme_ != ReconstructionScheme::FirstOrder &&
        scheme_ != ReconstructionScheme::TVD &&
        scheme_ != ReconstructionScheme::ENO &&
        scheme_ != ReconstructionScheme::WENO) {
        throw std::invalid_argument("Unsupported structured reconstruction scheme");
    }
    if (eno_order_ != 2 && eno_order_ != 3) {
        throw std::invalid_argument("StructuredFVSolver: eno_order must be 2 or 3");
    }
    if (weno_order_ != 3 && weno_order_ != 5) {
        throw std::invalid_argument("StructuredFVSolver: weno_order must be 3 or 5");
    }
    Index required_ghost = 1;
    if (scheme_ == ReconstructionScheme::TVD) {
        required_ghost = 2;
    } else if (scheme_ == ReconstructionScheme::ENO) {
        required_ghost = static_cast<Index>(eno_order_);
    } else if (scheme_ == ReconstructionScheme::WENO) {
        required_ghost = (weno_order_ == 3) ? 2 : 3;
    }
    if (mesh_.nghost() < required_ghost) {
        throw std::invalid_argument(
            "StructuredFVSolver: insufficient mesh ghost layers for reconstruction scheme");
    }
    if (decomp_ && decomp_->size > 1 &&
        (mesh_.nx() < required_ghost || mesh_.ny() < required_ghost)) {
        throw std::invalid_argument(
            "StructuredFVSolver: MPI subdomain smaller than required ghost layers");
    }
}

Real StructuredFVSolver::compute_rhs_eno3(const StructuredField2D& u,
                                          StructuredField2D& dudt) const {
    return compute_rhs_weno5(u, dudt);  // unified high-order 2-pass path below
}

Real StructuredFVSolver::compute_rhs_weno5(const StructuredField2D& u,
                                           StructuredField2D& dudt) const {
    const Index nx = mesh_.nx();
    const Index ny = mesh_.ny();
    const Real dx = mesh_.dx();
    const Real dy = mesh_.dy();
    const Real inv_dx = 1.0 / dx;
    const Real inv_dy = 1.0 / dy;
    const bool use_weno = (scheme_ == ReconstructionScheme::WENO);
    const bool use_weno5 = use_weno && (weno_order_ == 5);
    const bool use_weno3 = use_weno && (weno_order_ == 3);
    const SignedIndex pad = use_weno3 ? 1 : 2;

    const Real inv_2sqrt3 = 0.5 / std::sqrt(3.0);
    const Real xi_out3 = 0.1 * std::sqrt(15.0);
    const Real eps_x = (weno_type_ == WENOWeightType::WENO_Z)
        ? std::max(1.0e-36, use_weno3 ? (dx * dx) : (dx * dx * dx * dx))
        : 1.0e-6;
    const Real eps_y = (weno_type_ == WENOWeightType::WENO_Z)
        ? std::max(1.0e-36, use_weno3 ? (dy * dy) : (dy * dy * dy * dy))
        : 1.0e-6;

    auto eval_fx = [&](Real x, Real y, Real uL, Real uR) -> Real {
        const Real fL = law_.fx(x, y, uL);
        const Real fR = law_.fx(x, y, uR);
        const Real ax = law_.alpha_x(x, y, uL, uR);
        return flux::rusanov(fL, fR, uL, uR, ax);
    };
    auto eval_fy = [&](Real x, Real y, Real uL, Real uR) -> Real {
        const Real gL = law_.fy(x, y, uL);
        const Real gR = law_.fy(x, y, uR);
        const Real ay = law_.alpha_y(x, y, uL, uR);
        return flux::rusanov(gL, gR, uL, uR, ay);
    };

    const bool phys_left = !decomp_ || decomp_->on_physical_left();
    const bool phys_right = !decomp_ || decomp_->on_physical_right();
    const bool phys_bottom = !decomp_ || decomp_->on_physical_bottom();
    const bool phys_top = !decomp_ || decomp_->on_physical_top();
    Real bnd_flux = 0.0;

    // X-direction: 1D cell-face reconstruction along X for j in [-pad, ny - 1 + pad], then Gauss along Y
    const Index ny_pad = ny + static_cast<Index>(2 * pad);
    std::vector<Real> face_L((nx + 1) * ny_pad, 0.0);
    std::vector<Real> face_R((nx + 1) * ny_pad, 0.0);
    for (Index jt = 0; jt < ny_pad; ++jt) {
        const SignedIndex js = static_cast<SignedIndex>(jt) - pad;
        for (SignedIndex is = -1; is <= static_cast<SignedIndex>(nx); ++is) {
            Real u_left = 0.0;
            Real u_right = 0.0;
            if (use_weno5) {
                weno5_cell_faces(u(is - 2, js), u(is - 1, js), u(is, js),
                                 u(is + 1, js), u(is + 2, js), weno_type_, eps_x,
                                 u_left, u_right);
            } else if (use_weno3) {
                weno3_cell_faces(u(is - 1, js), u(is, js), u(is + 1, js),
                                 weno_type_, eps_x, u_left, u_right);
            } else {
                eno3_cell_faces(u(is - 2, js), u(is - 1, js), u(is, js),
                                u(is + 1, js), u(is + 2, js), u_left, u_right);
            }
            if (is + 1 <= static_cast<SignedIndex>(nx)) {
                face_L[static_cast<Index>(is + 1) * ny_pad + jt] = u_right;
            }
            if (is >= 0) {
                face_R[static_cast<Index>(is) * ny_pad + jt] = u_left;
            }
        }
    }

    const Real y_off2 = dy * inv_2sqrt3;
    const Real y_off3 = dy * xi_out3;
    constexpr Real w_g3_out = 5.0 / 18.0;
    constexpr Real w_g3_mid = 8.0 / 18.0;
    for (Index i = 0; i <= nx; ++i) {
        const SignedIndex is = static_cast<SignedIndex>(i);
        const Real* colL = &face_L[i * ny_pad];
        const Real* colR = &face_R[i * ny_pad];
        for (Index j = 0; j < ny; ++j) {
            const Index jt = j + static_cast<Index>(pad);
            const SignedIndex js = static_cast<SignedIndex>(j);
            const Vec2 xf = mesh_.x_face_center(i, j);
            Real F = 0.0;
            if (use_weno5) {
                Real uLm = 0.0, uL0 = 0.0, uLp = 0.0;
                Real uRm = 0.0, uR0 = 0.0, uRp = 0.0;
                weno5_gauss3(colL[jt - 2], colL[jt - 1], colL[jt], colL[jt + 1], colL[jt + 2],
                             weno_type_, eps_y, uLm, uL0, uLp);
                weno5_gauss3(colR[jt - 2], colR[jt - 1], colR[jt], colR[jt + 1], colR[jt + 2],
                             weno_type_, eps_y, uRm, uR0, uRp);
                F = w_g3_out * eval_fx(xf.x, xf.y - y_off3, uLm, uRm)
                  + w_g3_mid * eval_fx(xf.x, xf.y, uL0, uR0)
                  + w_g3_out * eval_fx(xf.x, xf.y + y_off3, uLp, uRp);
            } else if (use_weno3) {
                Real uLm = 0.0, uLp = 0.0, uRm = 0.0, uRp = 0.0;
                weno3_gauss2(colL[jt - 1], colL[jt], colL[jt + 1], weno_type_, eps_y, uLm, uLp);
                weno3_gauss2(colR[jt - 1], colR[jt], colR[jt + 1], weno_type_, eps_y, uRm, uRp);
                F = 0.5 * (eval_fx(xf.x, xf.y - y_off2, uLm, uRm)
                         + eval_fx(xf.x, xf.y + y_off2, uLp, uRp));
            } else {
                Real uLm = 0.0, uLp = 0.0, uRm = 0.0, uRp = 0.0;
                eno3_gauss2(colL[jt - 2], colL[jt - 1], colL[jt], colL[jt + 1], colL[jt + 2], uLm, uLp);
                eno3_gauss2(colR[jt - 2], colR[jt - 1], colR[jt], colR[jt + 1], colR[jt + 2], uRm, uRp);
                F = 0.5 * (eval_fx(xf.x, xf.y - y_off2, uLm, uRm)
                         + eval_fx(xf.x, xf.y + y_off2, uLp, uRp));
            }

            if (i > 0) {
                dudt(is - 1, js) -= F * inv_dx;
            } else if (phys_left) {
                bnd_flux -= F * dy;
            }
            if (i < nx) {
                dudt(is, js) += F * inv_dx;
            } else if (phys_right) {
                bnd_flux += F * dy;
            }
        }
    }

    // Y-direction: 1D cell-face reconstruction along Y for i in [-pad, nx - 1 + pad], then Gauss along X
    const Index nx_pad = nx + static_cast<Index>(2 * pad);
    face_L.assign((ny + 1) * nx_pad, 0.0);
    face_R.assign((ny + 1) * nx_pad, 0.0);
    for (Index it = 0; it < nx_pad; ++it) {
        const SignedIndex is = static_cast<SignedIndex>(it) - pad;
        for (SignedIndex js = -1; js <= static_cast<SignedIndex>(ny); ++js) {
            Real u_left = 0.0;
            Real u_right = 0.0;
            if (use_weno5) {
                weno5_cell_faces(u(is, js - 2), u(is, js - 1), u(is, js),
                                 u(is, js + 1), u(is, js + 2), weno_type_, eps_y,
                                 u_left, u_right);
            } else if (use_weno3) {
                weno3_cell_faces(u(is, js - 1), u(is, js), u(is, js + 1),
                                 weno_type_, eps_y, u_left, u_right);
            } else {
                eno3_cell_faces(u(is, js - 2), u(is, js - 1), u(is, js),
                                u(is, js + 1), u(is, js + 2), u_left, u_right);
            }
            if (js + 1 <= static_cast<SignedIndex>(ny)) {
                face_L[static_cast<Index>(js + 1) * nx_pad + it] = u_right;
            }
            if (js >= 0) {
                face_R[static_cast<Index>(js) * nx_pad + it] = u_left;
            }
        }
    }

    const Real x_off2 = dx * inv_2sqrt3;
    const Real x_off3 = dx * xi_out3;
    for (Index j = 0; j <= ny; ++j) {
        const SignedIndex js = static_cast<SignedIndex>(j);
        const Real* rowL = &face_L[j * nx_pad];
        const Real* rowR = &face_R[j * nx_pad];
        for (Index i = 0; i < nx; ++i) {
            const Index it = i + static_cast<Index>(pad);
            const SignedIndex is = static_cast<SignedIndex>(i);
            const Vec2 yf = mesh_.y_face_center(i, j);
            Real G = 0.0;
            if (use_weno5) {
                Real uLm = 0.0, uL0 = 0.0, uLp = 0.0;
                Real uRm = 0.0, uR0 = 0.0, uRp = 0.0;
                weno5_gauss3(rowL[it - 2], rowL[it - 1], rowL[it], rowL[it + 1], rowL[it + 2],
                             weno_type_, eps_x, uLm, uL0, uLp);
                weno5_gauss3(rowR[it - 2], rowR[it - 1], rowR[it], rowR[it + 1], rowR[it + 2],
                             weno_type_, eps_x, uRm, uR0, uRp);
                G = w_g3_out * eval_fy(yf.x - x_off3, yf.y, uLm, uRm)
                  + w_g3_mid * eval_fy(yf.x, yf.y, uL0, uR0)
                  + w_g3_out * eval_fy(yf.x + x_off3, yf.y, uLp, uRp);
            } else if (use_weno3) {
                Real uLm = 0.0, uLp = 0.0, uRm = 0.0, uRp = 0.0;
                weno3_gauss2(rowL[it - 1], rowL[it], rowL[it + 1], weno_type_, eps_x, uLm, uLp);
                weno3_gauss2(rowR[it - 1], rowR[it], rowR[it + 1], weno_type_, eps_x, uRm, uRp);
                G = 0.5 * (eval_fy(yf.x - x_off2, yf.y, uLm, uRm)
                         + eval_fy(yf.x + x_off2, yf.y, uLp, uRp));
            } else {
                Real uLm = 0.0, uLp = 0.0, uRm = 0.0, uRp = 0.0;
                eno3_gauss2(rowL[it - 2], rowL[it - 1], rowL[it], rowL[it + 1], rowL[it + 2], uLm, uLp);
                eno3_gauss2(rowR[it - 2], rowR[it - 1], rowR[it], rowR[it + 1], rowR[it + 2], uRm, uRp);
                G = 0.5 * (eval_fy(yf.x - x_off2, yf.y, uLm, uRm)
                         + eval_fy(yf.x + x_off2, yf.y, uLp, uRp));
            }

            if (j > 0) {
                dudt(is, js - 1) -= G * inv_dy;
            } else if (phys_bottom) {
                bnd_flux -= G * dx;
            }
            if (j < ny) {
                dudt(is, js) += G * inv_dy;
            } else if (phys_top) {
                bnd_flux += G * dx;
            }
        }
    }

    return bnd_flux;
}

void StructuredFVSolver::compute_rhs(StructuredField2D& u,
                                     Real t,
                                     StructuredField2D& dudt,
                                     Real* boundary_flux_out) const {
    if (decomp_) {
        mpi::exchange_halos_and_apply_bcs(mesh_, u, bc_, *decomp_, t, &halo_scratch_);
    } else {
        mesh_.apply_boundary_conditions(u, bc_, t);
    }
    dudt.fill(0.0);

    if ((scheme_ == ReconstructionScheme::ENO && eno_order_ == 3) ||
        scheme_ == ReconstructionScheme::WENO) {
        Real bnd_flux = compute_rhs_weno5(u, dudt);
        if (decomp_) {
            bnd_flux = decomp_->allreduce_sum(bnd_flux);
        }
        if (boundary_flux_out) {
            *boundary_flux_out = bnd_flux;
        }
        return;
    }

    const Index nx = mesh_.nx();
    const Index ny = mesh_.ny();
    const Real dx = mesh_.dx();
    const Real dy = mesh_.dy();
    const Real inv_dx = 1.0 / dx;
    const Real inv_dy = 1.0 / dy;
    const bool use_tvd = (scheme_ == ReconstructionScheme::TVD);
    const bool use_eno2 = (scheme_ == ReconstructionScheme::ENO && eno_order_ == 2);

    const bool phys_left = !decomp_ || decomp_->on_physical_left();
    const bool phys_right = !decomp_ || decomp_->on_physical_right();
    const bool phys_bottom = !decomp_ || decomp_->on_physical_bottom();
    const bool phys_top = !decomp_ || decomp_->on_physical_top();
    Real bnd_flux = 0.0;

    // X-direction face fluxes (faces i - 1/2 for i = 0 .. nx)
    for (Index j = 0; j < ny; ++j) {
        const SignedIndex js = static_cast<SignedIndex>(j);
        for (Index i = 0; i <= nx; ++i) {
            const SignedIndex is = static_cast<SignedIndex>(i);
            const Vec2 xf = mesh_.x_face_center(i, j);
            Real uL = u(is - 1, js);
            Real uR = u(is, js);
            if (use_tvd) {
                apply_muscl_tvd(uL, uR, u(is - 2, js), u(is + 1, js), limiter_);
            } else if (use_eno2) {
                apply_eno2(uL, uR, u(is - 2, js), u(is + 1, js));
            }
            const Real fL = law_.fx(xf.x, xf.y, uL);
            const Real fR = law_.fx(xf.x, xf.y, uR);
            const Real ax = law_.alpha_x(xf.x, xf.y, uL, uR);
            const Real F = flux::rusanov(fL, fR, uL, uR, ax);

            if (i > 0) {
                dudt(is - 1, js) -= F * inv_dx;
            } else if (phys_left) {
                bnd_flux -= F * dy;  // outward normal at x_min is (-1, 0)
            }
            if (i < nx) {
                dudt(is, js) += F * inv_dx;
            } else if (phys_right) {
                bnd_flux += F * dy;  // outward normal at x_max is (+1, 0)
            }
        }
    }

    // Y-direction face fluxes (faces j - 1/2 for j = 0 .. ny)
    for (Index j = 0; j <= ny; ++j) {
        const SignedIndex js = static_cast<SignedIndex>(j);
        for (Index i = 0; i < nx; ++i) {
            const SignedIndex is = static_cast<SignedIndex>(i);
            const Vec2 yf = mesh_.y_face_center(i, j);
            Real uL = u(is, js - 1);
            Real uR = u(is, js);
            if (use_tvd) {
                apply_muscl_tvd(uL, uR, u(is, js - 2), u(is, js + 1), limiter_);
            } else if (use_eno2) {
                apply_eno2(uL, uR, u(is, js - 2), u(is, js + 1));
            }
            const Real gL = law_.fy(yf.x, yf.y, uL);
            const Real gR = law_.fy(yf.x, yf.y, uR);
            const Real ay = law_.alpha_y(yf.x, yf.y, uL, uR);
            const Real G = flux::rusanov(gL, gR, uL, uR, ay);

            if (j > 0) {
                dudt(is, js - 1) -= G * inv_dy;
            } else if (phys_bottom) {
                bnd_flux -= G * dx;  // outward normal at y_min is (0, -1)
            }
            if (j < ny) {
                dudt(is, js) += G * inv_dy;
            } else if (phys_top) {
                bnd_flux += G * dx;  // outward normal at y_max is (0, +1)
            }
        }
    }

    if (decomp_) {
        bnd_flux = decomp_->allreduce_sum(bnd_flux);
    }
    if (boundary_flux_out) {
        *boundary_flux_out = bnd_flux;
    }
}

Real StructuredFVSolver::compute_dt(const StructuredField2D& u, Real cfl) const {
    Real max_ax = 0.0;
    Real max_ay = 0.0;
    law_.max_speeds(u, max_ax, max_ay);
    if (decomp_) {
        max_ax = decomp_->allreduce_max(max_ax);
        max_ay = decomp_->allreduce_max(max_ay);
    }
    const Real inv_dt_bound = max_ax / mesh_.dx() + max_ay / mesh_.dy();
    if (inv_dt_bound <= 1.0e-30) {
        return 1.0e-3;
    }
    return cfl / inv_dt_bound;
}

SolverRunSummary StructuredFVSolver::run(
    StructuredField2D& u,
    const SolverRunOptions& opts,
    ExactDiagFn exact_diag) const {
    const auto t_start = std::chrono::steady_clock::now();

    SolverRunSummary summary;
    summary.initial_stats = u.interior_stats(mesh_.cell_area());
    if (decomp_) {
        summary.initial_stats = decomp_->allreduce_stats(summary.initial_stats);
    }

    const int my_rank = decomp_ ? decomp_->rank : 0;
    const bool want_vtk = !opts.vtk_dir.empty();
    io::PvdWriter pvd(opts.vtk_dir + "/" + opts.pvd_name);
    Index snap_idx = 0;

    StructuredMesh2D vtk_mesh = mesh_;
    if (want_vtk && decomp_ && decomp_->size > 1 && my_rank == 0) {
        const Real gx0 = mesh_.xmin() - static_cast<Real>(decomp_->i0) * mesh_.dx();
        const Real gx1 = gx0 + static_cast<Real>(decomp_->nx_global) * mesh_.dx();
        const Real gy0 = mesh_.ymin() - static_cast<Real>(decomp_->j0) * mesh_.dy();
        const Real gy1 = gy0 + static_cast<Real>(decomp_->ny_global) * mesh_.dy();
        vtk_mesh = StructuredMesh2D(
            gx0, gx1, gy0, gy1, decomp_->nx_global, decomp_->ny_global, 0);
    }

    StructuredField2D u_exact, u_err;
    if (want_vtk && exact_diag) {
        u_exact = mesh_.create_field();
        u_err = mesh_.create_field();
    }

    auto write_snapshot = [&](Real t_now) {
        if (!want_vtk) return;
        std::ostringstream fname;
        fname << "step_" << std::setw(4) << std::setfill('0') << snap_idx++ << ".vti";
        const std::string full_path = opts.vtk_dir + "/" + fname.str();
        if (exact_diag) {
            exact_diag(t_now, mesh_, u, u_exact, u_err);
        }
        if (decomp_ && decomp_->size > 1) {
            const auto ug = mpi::gather_structured_field(u, *decomp_);
            StructuredField2D ex_g, err_g;
            if (exact_diag) {
                ex_g = mpi::gather_structured_field(u_exact, *decomp_);
                err_g = mpi::gather_structured_field(u_err, *decomp_);
            }
            if (my_rank == 0) {
                if (exact_diag) {
                    io::write_vti(full_path, vtk_mesh,
                                  {{"u", &ug}, {"u_exact", &ex_g}, {"error", &err_g}});
                } else {
                    io::write_vti(full_path, vtk_mesh, {{"u", &ug}});
                }
                pvd.add_step(t_now, fname.str());
            }
        } else if (my_rank == 0) {
            if (exact_diag) {
                io::write_vti(full_path, mesh_,
                              {{"u", &u}, {"u_exact", &u_exact}, {"error", &u_err}});
            } else {
                io::write_vti(full_path, mesh_, {{"u", &u}});
            }
            pvd.add_step(t_now, fname.str());
        }
    };

    Real t = 0.0;
    Real next_output_t = (opts.output_interval > 0.0) ? opts.output_interval : opts.t_end + 1.0;
    write_snapshot(t);

    const std::vector<Real> stage_weights = time::rk_stage_weights(opts.rk_method);
    Index stage_idx = 0;
    Real step_bnd_flux = 0.0;

    time::RKStepper<StructuredField2D> stepper(opts.rk_method);
    auto rhs_fn = [&](StructuredField2D& state, Real stage_t, StructuredField2D& dudt) {
        Real bnd_flux = 0.0;
        this->compute_rhs(state, stage_t, dudt, &bnd_flux);
        step_bnd_flux += stage_weights.at(stage_idx++) * bnd_flux;
    };

    auto compute_mass = [&](const StructuredField2D& field) {
        Real sum = 0.0;
        for (Index j = 0; j < mesh_.ny(); ++j) {
            for (Index i = 0; i < mesh_.nx(); ++i) {
                sum += field(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j));
            }
        }
        Real mass = sum * mesh_.cell_area();
        if (decomp_) {
            mass = decomp_->allreduce_sum(mass);
        }
        return mass;
    };

    Real mass_prev = summary.initial_stats.total_mass;

    while (t < opts.t_end - 1.0e-13) {
        Real dt = compute_dt(u, opts.cfl);
        if (t + dt > opts.t_end) {
            dt = opts.t_end - t;
        }
        stage_idx = 0;
        step_bnd_flux = 0.0;
        stepper.step(u, t, dt, rhs_fn);
        t += dt;
        summary.steps += 1;

        const Real mass_now = compute_mass(u);
        const Real step_res = std::abs((mass_now - mass_prev) + dt * step_bnd_flux);
        summary.max_conservation_residual = std::max(summary.max_conservation_residual, step_res);
        summary.net_boundary_outflow += dt * step_bnd_flux;
        mass_prev = mass_now;

        if (opts.output_interval > 0.0 && t >= next_output_t - 1.0e-12 && t < opts.t_end - 1.0e-12) {
            write_snapshot(t);
            while (next_output_t <= t + 1.0e-12) {
                next_output_t += opts.output_interval;
            }
        }
    }

    if (snap_idx == 1 || opts.output_interval > 0.0) {
        write_snapshot(t);
    }
    if (want_vtk && my_rank == 0) {
        pvd.save();
    }

    const auto t_stop = std::chrono::steady_clock::now();
    summary.final_time = t;
    summary.runtime_sec = std::chrono::duration<Real>(t_stop - t_start).count();
    summary.final_stats = u.interior_stats(mesh_.cell_area());
    if (decomp_) {
        summary.final_stats = decomp_->allreduce_stats(summary.final_stats);
    }
    summary.cumulative_conservation_residual = std::abs(
        (summary.final_stats.total_mass - summary.initial_stats.total_mass) +
        summary.net_boundary_outflow);
    return summary;
}

}  // namespace fvm::solvers
