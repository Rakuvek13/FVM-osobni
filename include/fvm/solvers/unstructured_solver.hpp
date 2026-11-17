#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/linalg/dense.hpp"
#include "fvm/mesh/unstructured_mesh.hpp"
#include "fvm/numerics/time_integrator.hpp"
#include "fvm/solvers/structured_solver.hpp"  // ReconstructionScheme

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fvm::solvers {

enum class UnstructuredLimiter {
    BarthJespersen,
    Venkatakrishnan
};

inline std::string to_string(UnstructuredLimiter l) {
    switch (l) {
        case UnstructuredLimiter::BarthJespersen:  return "barth-jespersen";
        case UnstructuredLimiter::Venkatakrishnan: return "venkatakrishnan";
    }
    return "unknown";
}

inline UnstructuredLimiter parse_unstructured_limiter(const std::string& s) {
    if (s == "barth-jespersen" || s == "barth_jespersen" || s == "bj") {
        return UnstructuredLimiter::BarthJespersen;
    }
    if (s == "venkatakrishnan" || s == "venkat" || s == "vk") {
        return UnstructuredLimiter::Venkatakrishnan;
    }
    throw std::invalid_argument("Unknown unstructured limiter: " + s);
}

struct UnstructuredRunOptions {
    ReconstructionScheme scheme = ReconstructionScheme::FirstOrder;
    UnstructuredLimiter limiter = UnstructuredLimiter::BarthJespersen;
    int eno_order = 3;
    WENOWeightType weno_type = WENOWeightType::WENO_Z;
    time::RKMethod rk_method = time::RKMethod::Euler;
    Real cfl = 0.5;
    Real t_end = 1.0;
    Real output_interval = 0.0;
    std::string vtk_dir;
    std::string pvd_name = "solution.pvd";
};

struct UnstructuredRunSummary {
    Index steps = 0;
    Real final_time = 0.0;
    Real dt_used = 0.0;
    Real max_discrete_divergence = 0.0;
    Real max_conservation_residual = 0.0;
    Real cumulative_conservation_residual = 0.0;
    Real net_boundary_outflow = 0.0;
    Real outlet_outflow_mass = 0.0;
    Real runtime_sec = 0.0;
    FieldStats initial_stats;
    FieldStats final_stats;
};

/// 2D unstructured finite-volume solver for passive scalar transport:
///   u_t + div(v(x, y) u) = 0
class UnstructuredAdvectionSolver {
public:
    using VelocityFn = std::function<Vec2(Real x, Real y)>;
    using InflowFn = std::function<Real(Real x, Real y, Real t, int boundary_tag)>;

    UnstructuredAdvectionSolver(
        const UnstructuredMesh2D& mesh,
        VelocityFn velocity,
        InflowFn inflow_bc = [](Real, Real, Real, int) { return 0.0; },
        ReconstructionScheme scheme = ReconstructionScheme::FirstOrder,
        int edge_quad_pts = 3,
        UnstructuredLimiter limiter = UnstructuredLimiter::BarthJespersen,
        int eno_order = 3,
        WENOWeightType weno_type = WENOWeightType::WENO_Z,
        const mpi::UnstructuredDecomp* decomp = nullptr);

    ReconstructionScheme scheme() const { return scheme_; }
    UnstructuredLimiter limiter() const { return limiter_; }
    int eno_order() const { return eno_order_; }
    WENOWeightType weno_type() const { return weno_type_; }
    const mpi::UnstructuredDecomp* mpi_decomp() const { return decomp_; }

    /// Maximum discrete divergence |sum_{e in dK} int_e v.n ds| / |K| over all cells.
    Real max_discrete_divergence() const { return max_discrete_div_; }

    /// Maximum explicit Forward-Euler / SSP positivity step dt_max = min_K |K| / sum_e (v.n |e|)^+.
    Real dt_cfl_limit() const { return dt_cfl_limit_; }

    void compute_rhs(const std::vector<Real>& u, Real t, std::vector<Real>& dudt,
                     Real* boundary_flux_out = nullptr,
                     Real* outlet_flux_out = nullptr) const;

    using ExactDiagFn = std::function<void(
        Real t, const UnstructuredMesh2D&, const std::vector<Real>&,
        std::vector<Real>& exact_out, std::vector<Real>& err_out)>;

    UnstructuredRunSummary run(
        std::vector<Real>& u,
        const UnstructuredRunOptions& opts,
        ExactDiagFn exact_diag = {}) const;

private:
    struct FaceQuadData {
        Real x = 0.0;
        Real y = 0.0;
        Real weighted_flux = 0.0;  // q_weight * face.length * (v(x, y) . normal)
    };

    struct CellGradStencil {
        std::vector<Index> neighbors;
        std::vector<Real> inv_distances;
        linalg::QRDecomposition qr;
    };

    struct SimplexPair2 {
        Index n0 = 0;
        Index n1 = 0;
        Real inv00 = 0.0, inv01 = 0.0;
        Real inv10 = 0.0, inv11 = 0.0;
    };

    struct P2Stencil {
        std::vector<Index> neighbors;
        std::vector<Real> weights;
        linalg::QRDecomposition qr;
    };

    struct CellENO3Data {
        Real inv_h = 1.0;
        Real m20 = 0.0, m11 = 0.0, m02 = 0.0;      // physical central moments
        Real m20_hat = 0.0, m11_hat = 0.0, m02_hat = 0.0;  // m_ab / area
        P2Stencil preferred;
        std::vector<P2Stencil> sectors;
    };

    struct CellPoly2 {
        Real gx = 0.0, gy = 0.0;
        Real qxx = 0.0, qxy = 0.0, qyy = 0.0;
        Real m20 = 0.0, m11 = 0.0, m02 = 0.0;

        Real eval_offset(const Vec2& dx) const {
            return gx * dx.x + gy * dx.y
                 + qxx * (dx.x * dx.x - m20)
                 + qxy * (dx.x * dx.y - m11)
                 + qyy * (dx.y * dx.y - m02);
        }
    };

    void compute_limited_gradients(const std::vector<Real>& u,
                                   std::vector<CellPoly2>& polys) const;
    void compute_eno2_polys(const std::vector<Real>& u,
                            std::vector<CellPoly2>& polys) const;
    void compute_p2_polys(const std::vector<Real>& u,
                          bool use_weno,
                          std::vector<CellPoly2>& polys) const;

    const UnstructuredMesh2D& mesh_;
    VelocityFn velocity_;
    InflowFn inflow_bc_;
    ReconstructionScheme scheme_;
    int edge_quad_pts_;
    UnstructuredLimiter limiter_;
    int eno_order_;
    WENOWeightType weno_type_;
    const mpi::UnstructuredDecomp* decomp_ = nullptr;

    std::vector<Index> active_cells_;
    std::vector<Index> active_faces_;
    mpi::UnstructuredCommSchedule u_sched_;
    mpi::UnstructuredCommSchedule poly_sched_;
    mutable std::vector<Real> u_work_;

    std::vector<Real> face_flux_rate_;       // int_e (v . n) ds per face
    std::vector<FaceQuadData> face_quads_;   // size = faces.size() * edge_quad_pts_
    std::vector<Vec2> cell_velocity_;        // velocity at cell centroid (for ParaView visualization)
    std::vector<CellGradStencil> stencils_;  // precomputed for TVD & ENO2 preferred
    std::vector<std::vector<SimplexPair2>> eno2_pairs_;
    std::vector<CellENO3Data> eno3_data_;
    Real max_discrete_div_ = 0.0;
    Real dt_cfl_limit_ = 0.0;
};

}  // namespace fvm::solvers
