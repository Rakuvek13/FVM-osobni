#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/numerics/time_integrator.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fvm::solvers {

enum class ReconstructionScheme {
    FirstOrder,
    TVD,
    ENO,
    WENO
};

inline std::string to_string(ReconstructionScheme s) {
    switch (s) {
        case ReconstructionScheme::FirstOrder: return "1st-Order";
        case ReconstructionScheme::TVD:        return "2nd-Order TVD";
        case ReconstructionScheme::ENO:        return "ENO";
        case ReconstructionScheme::WENO:       return "WENO";
    }
    return "Unknown";
}

enum class WENOWeightType {
    WENO_Z,
    WENO_JS
};

inline std::string to_string(WENOWeightType w) {
    switch (w) {
        case WENOWeightType::WENO_Z:  return "z";
        case WENOWeightType::WENO_JS: return "js";
    }
    return "unknown";
}

inline WENOWeightType parse_weno_type(const std::string& s) {
    if (s == "z" || s == "weno-z" || s == "wenoz") return WENOWeightType::WENO_Z;
    if (s == "js" || s == "weno-js" || s == "wenojs") return WENOWeightType::WENO_JS;
    throw std::invalid_argument("Unknown WENO weight type: " + s);
}

enum class TVDSlopeLimiter {
    Minmod,
    VanLeer,
    MC,
    Superbee
};

inline std::string to_string(TVDSlopeLimiter l) {
    switch (l) {
        case TVDSlopeLimiter::Minmod:   return "minmod";
        case TVDSlopeLimiter::VanLeer:  return "vanleer";
        case TVDSlopeLimiter::MC:       return "mc";
        case TVDSlopeLimiter::Superbee: return "superbee";
    }
    return "unknown";
}

inline TVDSlopeLimiter parse_tvd_limiter(const std::string& s) {
    if (s == "minmod") return TVDSlopeLimiter::Minmod;
    if (s == "vanleer" || s == "van_leer") return TVDSlopeLimiter::VanLeer;
    if (s == "mc") return TVDSlopeLimiter::MC;
    if (s == "superbee") return TVDSlopeLimiter::Superbee;
    throw std::invalid_argument("Unknown TVD slope limiter: " + s);
}

inline std::vector<ReconstructionScheme> parse_scheme_selection(const std::string& s) {
    if (s == "first" || s == "1st") return {ReconstructionScheme::FirstOrder};
    if (s == "tvd" || s == "2nd") return {ReconstructionScheme::TVD};
    if (s == "eno") return {ReconstructionScheme::ENO};
    if (s == "weno") return {ReconstructionScheme::WENO};
    if (s == "all") {
        return {ReconstructionScheme::FirstOrder,
                ReconstructionScheme::TVD,
                ReconstructionScheme::ENO,
                ReconstructionScheme::WENO};
    }
    throw std::invalid_argument("Unknown scheme selection: " + s);
}

/// Definition of a 2D scalar conservation law:
///   u_t + d_x f(x, y, u) + d_y g(x, y, u) = 0
struct StructuredConservationLaw {
    std::function<Real(Real x, Real y, Real u)> fx;
    std::function<Real(Real x, Real y, Real u)> fy;

    /// Upper bound on |d f / d u| over [min(uL,uR), max(uL,uR)] at face (x, y).
    std::function<Real(Real x, Real y, Real uL, Real uR)> alpha_x;
    /// Upper bound on |d g / d u| over [min(uL,uR), max(uL,uR)] at face (x, y).
    std::function<Real(Real x, Real y, Real uL, Real uR)> alpha_y;

    /// Global max wave speed estimator over current field (used for dt CFL step).
    std::function<void(const StructuredField2D& u, Real& max_ax, Real& max_ay)> max_speeds;
};

/// Builds a linear advection conservation law u_t + d_x(a_x u) + d_y(a_y u) = 0
/// using exact upwind flux (Rusanov with alpha = |a| equals upwind for linear flux).
StructuredConservationLaw make_linear_advection_law(Real ax, Real ay);

struct SolverRunOptions {
    ReconstructionScheme scheme = ReconstructionScheme::FirstOrder;
    TVDSlopeLimiter tvd_limiter = TVDSlopeLimiter::MC;
    int eno_order = 3;
    WENOWeightType weno_type = WENOWeightType::WENO_Z;
    time::RKMethod rk_method = time::RKMethod::Euler;
    Real cfl = 0.45;
    Real t_end = 1.0;
    Real output_interval = 0.0;  // 0 => save initial & final only
    std::string vtk_dir;
    std::string pvd_name = "solution.pvd";
};

struct SolverRunSummary {
    Index steps = 0;
    Real final_time = 0.0;
    Real runtime_sec = 0.0;
    Real max_conservation_residual = 0.0;         // max_n |R_M^n|
    Real cumulative_conservation_residual = 0.0;  // |(M(T) - M(0)) + int_0^T flux_dOmega|
    Real net_boundary_outflow = 0.0;              // time-integrated outward boundary flux
    FieldStats initial_stats;
    FieldStats final_stats;
};

/// 2D structured finite-volume solver for scalar conservation laws.
class StructuredFVSolver {
public:
    StructuredFVSolver(const StructuredMesh2D& mesh,
                       StructuredBoundaryConfig bc,
                       StructuredConservationLaw law,
                       ReconstructionScheme scheme = ReconstructionScheme::FirstOrder,
                       TVDSlopeLimiter limiter = TVDSlopeLimiter::MC,
                       int eno_order = 3,
                       WENOWeightType weno_type = WENOWeightType::WENO_Z,
                       const mpi::CartDecomp2D* decomp = nullptr);

    ReconstructionScheme scheme() const { return scheme_; }
    TVDSlopeLimiter limiter() const { return limiter_; }
    int eno_order() const { return eno_order_; }
    WENOWeightType weno_type() const { return weno_type_; }
    const mpi::CartDecomp2D* mpi_decomp() const { return decomp_; }

    /// Evaluates du/dt = L(u, t) after enforcing boundary conditions on u at time t.
    /// Optionally returns outward boundary flux integral sum_{e in dOmega} |e| H_e.
    void compute_rhs(StructuredField2D& u, Real t, StructuredField2D& dudt,
                     Real* boundary_flux_out = nullptr) const;

    /// Computes stable time step dt <= CFL / (max_ax / dx + max_ay / dy).
    Real compute_dt(const StructuredField2D& u, Real cfl) const;

    using ExactDiagFn = std::function<void(
        Real t, const StructuredMesh2D&, const StructuredField2D&,
        StructuredField2D& exact_out, StructuredField2D& err_out)>;

    /// Integrates u from t = 0 to opts.t_end, optionally writing ParaView .vti + .pvd files.
    SolverRunSummary run(
        StructuredField2D& u,
        const SolverRunOptions& opts,
        ExactDiagFn exact_diag = {}) const;

private:
    Real compute_rhs_eno3(const StructuredField2D& u, StructuredField2D& dudt) const;
    Real compute_rhs_weno5(const StructuredField2D& u, StructuredField2D& dudt) const;

    const StructuredMesh2D& mesh_;
    StructuredBoundaryConfig bc_;
    StructuredConservationLaw law_;
    ReconstructionScheme scheme_;
    TVDSlopeLimiter limiter_;
    int eno_order_;
    WENOWeightType weno_type_;
    const mpi::CartDecomp2D* decomp_ = nullptr;
    mutable mpi::HaloScratch halo_scratch_;
};

}  // namespace fvm::solvers
