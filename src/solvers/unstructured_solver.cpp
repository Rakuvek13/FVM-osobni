#include "fvm/solvers/unstructured_solver.hpp"
#include "fvm/io/vtk_writer.hpp"
#include "fvm/numerics/quadrature.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fvm::solvers {

UnstructuredAdvectionSolver::UnstructuredAdvectionSolver(
    const UnstructuredMesh2D& mesh,
    VelocityFn velocity,
    InflowFn inflow_bc,
    ReconstructionScheme scheme,
    int edge_quad_pts,
    UnstructuredLimiter limiter,
    int eno_order,
    WENOWeightType weno_type,
    const mpi::UnstructuredDecomp* decomp)
    : mesh_(mesh),
      velocity_(std::move(velocity)),
      inflow_bc_(std::move(inflow_bc)),
      scheme_(scheme),
      edge_quad_pts_(edge_quad_pts),
      limiter_(limiter),
      eno_order_(eno_order),
      weno_type_(weno_type),
      decomp_(decomp) {
    if (mesh_.num_cells() == 0) {
        throw std::invalid_argument("UnstructuredAdvectionSolver: empty mesh");
    }
    if (scheme_ != ReconstructionScheme::FirstOrder &&
        scheme_ != ReconstructionScheme::TVD &&
        scheme_ != ReconstructionScheme::ENO &&
        scheme_ != ReconstructionScheme::WENO) {
        throw std::runtime_error("Unsupported unstructured reconstruction scheme");
    }
    if (eno_order_ != 2 && eno_order_ != 3) {
        throw std::invalid_argument("UnstructuredAdvectionSolver: eno_order must be 2 or 3");
    }

    const auto q1d = quadrature::gauss_legendre_01(edge_quad_pts_);
    const auto& verts = mesh_.vertices();
    const auto& faces = mesh_.faces();
    const auto& cells = mesh_.cells();
    const Index nq = static_cast<Index>(q1d.size());

    const bool use_mpi = (decomp_ != nullptr && decomp_->size > 1);
    const int my_rank = use_mpi ? decomp_->rank : 0;
    if (use_mpi) {
        active_cells_ = decomp_->owned_cells;
        for (Index f = 0; f < faces.size(); ++f) {
            const auto& fc = faces[f];
            if (decomp_->owner[fc.left_cell] == my_rank ||
                (fc.right_cell >= 0 && decomp_->owner[static_cast<Index>(fc.right_cell)] == my_rank)) {
                active_faces_.push_back(f);
            }
        }
    } else {
        active_cells_.resize(cells.size());
        std::iota(active_cells_.begin(), active_cells_.end(), 0);
        active_faces_.resize(faces.size());
        std::iota(active_faces_.begin(), active_faces_.end(), 0);
    }

    face_flux_rate_.resize(faces.size(), 0.0);
    face_quads_.resize(faces.size() * nq);
    for (Index f = 0; f < faces.size(); ++f) {
        const auto& face = faces[f];
        const Vec2& p0 = verts[face.v0];
        const Vec2& p1 = verts[face.v1];
        Real flux_sum = 0.0;
        for (Index qi = 0; qi < nq; ++qi) {
            const auto& qp = q1d[qi];
            const Vec2 x = (1.0 - qp.xi) * p0 + qp.xi * p1;
            const Real wflux = qp.weight * face.length * dot(velocity_(x.x, x.y), face.normal);
            face_quads_[f * nq + qi] = {x.x, x.y, wflux};
            flux_sum += wflux;
        }
        face_flux_rate_[f] = flux_sum;
    }

    const bool need_linear_stencils =
        (scheme_ == ReconstructionScheme::TVD) ||
        (scheme_ == ReconstructionScheme::ENO && eno_order_ == 2);
    if (need_linear_stencils) {
        stencils_.resize(cells.size());
        for (Index c : active_cells_) {
            const std::vector<Index> raw = mesh_.collect_stencil(c, 6, true);
            CellGradStencil& st = stencils_[c];
            st.neighbors.reserve(raw.size());
            for (Index nb : raw) {
                if (nb != c) {
                    st.neighbors.push_back(nb);
                }
            }
            if (st.neighbors.size() < 2) {
                throw std::runtime_error(
                    "UnstructuredAdvectionSolver: need at least 2 neighbor cells in stencil");
            }
            const Index m = static_cast<Index>(st.neighbors.size());
            linalg::DenseMatrix A(m, 2);
            st.inv_distances.resize(m);
            const Vec2& xc = cells[c].centroid;
            for (Index r = 0; r < m; ++r) {
                const Vec2 d = cells[st.neighbors[r]].centroid - xc;
                const Real dist = std::max(d.norm(), 1.0e-14);
                const Real inv_d = 1.0 / dist;
                st.inv_distances[r] = inv_d;
                A(r, 0) = d.x * inv_d;
                A(r, 1) = d.y * inv_d;
            }
            st.qr = linalg::QRDecomposition::compute(std::move(A));
        }
    }

    if (scheme_ == ReconstructionScheme::ENO && eno_order_ == 2) {
        eno2_pairs_.resize(cells.size());
        for (Index c : active_cells_) {
            const Vec2& xc = cells[c].centroid;
            std::vector<Index> cand = stencils_[c].neighbors;
            std::sort(cand.begin(), cand.end(), [&](Index a, Index b) {
                return (cells[a].centroid - xc).norm2() < (cells[b].centroid - xc).norm2();
            });
            const Index cap = std::min<Index>(cand.size(), 8);
            for (Index a = 0; a < cap; ++a) {
                const Vec2 da = cells[cand[a]].centroid - xc;
                const Real la = da.norm();
                if (la <= 1.0e-14) continue;
                for (Index b = a + 1; b < cap; ++b) {
                    const Vec2 db = cells[cand[b]].centroid - xc;
                    const Real lb = db.norm();
                    if (lb <= 1.0e-14) continue;
                    const Real det = da.x * db.y - da.y * db.x;
                    if (std::abs(det) / (la * lb) < 0.35) continue;
                    const Real inv_det = 1.0 / det;
                    eno2_pairs_[c].push_back({
                        cand[a], cand[b],
                        db.y * inv_det, -da.y * inv_det,
                        -db.x * inv_det, da.x * inv_det
                    });
                    if (eno2_pairs_[c].size() >= 10) break;
                }
                if (eno2_pairs_[c].size() >= 10) break;
            }
        }
    } else if ((scheme_ == ReconstructionScheme::ENO && eno_order_ == 3) ||
               scheme_ == ReconstructionScheme::WENO) {
        const auto qtri = quadrature::triangle_rule(5);
        std::vector<Real> m20_all(cells.size(), 0.0);
        std::vector<Real> m11_all(cells.size(), 0.0);
        std::vector<Real> m02_all(cells.size(), 0.0);
        for (Index c = 0; c < cells.size(); ++c) {
            const Vec2& xc = cells[c].centroid;
            const auto& vids = cells[c].vertex_ids;
            if (vids.size() == 3) {
                const Vec2& v0 = verts[vids[0]];
                const Vec2& v1 = verts[vids[1]];
                const Vec2& v2 = verts[vids[2]];
                for (const auto& qp : qtri) {
                    const Vec2 x = qp.lambda1 * v0 + qp.lambda2 * v1 + qp.lambda3 * v2;
                    const Vec2 d = x - xc;
                    m20_all[c] += qp.weight * d.x * d.x;
                    m11_all[c] += qp.weight * d.x * d.y;
                    m02_all[c] += qp.weight * d.y * d.y;
                }
            } else {
                Real total_a = 0.0;
                for (Index k = 0; k < vids.size(); ++k) {
                    const Vec2& va = verts[vids[k]];
                    const Vec2& vb = verts[vids[(k + 1) % vids.size()]];
                    const Real sub_a = 0.5 * std::abs((va.x - xc.x) * (vb.y - xc.y) -
                                                      (va.y - xc.y) * (vb.x - xc.x));
                    total_a += sub_a;
                    for (const auto& qp : qtri) {
                        const Vec2 x = qp.lambda1 * xc + qp.lambda2 * va + qp.lambda3 * vb;
                        const Vec2 d = x - xc;
                        m20_all[c] += sub_a * qp.weight * d.x * d.x;
                        m11_all[c] += sub_a * qp.weight * d.x * d.y;
                        m02_all[c] += sub_a * qp.weight * d.y * d.y;
                    }
                }
                if (total_a > 0.0) {
                    m20_all[c] /= total_a;
                    m11_all[c] /= total_a;
                    m02_all[c] /= total_a;
                }
            }
        }

        eno3_data_.resize(cells.size());
        for (Index c : active_cells_) {
            CellENO3Data& ed = eno3_data_[c];
            const Real area_c = cells[c].area;
            const Real h_c = std::sqrt(area_c);
            const Real inv_h = 1.0 / h_c;
            const Real inv_h2 = inv_h * inv_h;
            ed.inv_h = inv_h;
            ed.m20 = m20_all[c];
            ed.m11 = m11_all[c];
            ed.m02 = m02_all[c];
            ed.m20_hat = m20_all[c] * inv_h2;
            ed.m11_hat = m11_all[c] * inv_h2;
            ed.m02_hat = m02_all[c] * inv_h2;

            const Vec2& xc = cells[c].centroid;
            std::vector<Index> cloud;
            const std::vector<Index> raw = mesh_.collect_stencil(c, 18, true);
            cloud.reserve(raw.size());
            for (Index nb : raw) {
                if (nb != c) cloud.push_back(nb);
            }
            std::sort(cloud.begin(), cloud.end(), [&](Index a, Index b) {
                return (cells[a].centroid - xc).norm2() < (cells[b].centroid - xc).norm2();
            });

            auto try_build_p2 = [&](const std::vector<Index>& nbs, Real max_cond, P2Stencil& out) {
                if (nbs.size() < 5) return false;
                const Index m = static_cast<Index>(nbs.size());
                linalg::DenseMatrix A(m, 5);
                out.neighbors = nbs;
                out.weights.resize(m);
                for (Index r = 0; r < m; ++r) {
                    const Index j = nbs[r];
                    const Vec2 d = cells[j].centroid - xc;
                    const Real dist_hat = d.norm() * inv_h;
                    const Real w = 1.0 / std::max(Real(1.0), dist_hat);
                    out.weights[r] = w;
                    const Real dx_h = d.x * inv_h;
                    const Real dy_h = d.y * inv_h;
                    A(r, 0) = w * dx_h;
                    A(r, 1) = w * dy_h;
                    A(r, 2) = w * ((m20_all[j] - ed.m20) * inv_h2 + dx_h * dx_h);
                    A(r, 3) = w * ((m11_all[j] - ed.m11) * inv_h2 + dx_h * dy_h);
                    A(r, 4) = w * ((m02_all[j] - ed.m02) * inv_h2 + dy_h * dy_h);
                }
                if (!linalg::QRDecomposition::try_compute(std::move(A), out.qr)) {
                    return false;
                }
                const Real dmin = out.qr.min_abs_r_diag();
                const Real dmax = out.qr.max_abs_r_diag();
                return dmin > 1.0e-3 && (dmax / dmin) < max_cond;
            };

            bool ok_pref = false;
            for (Index count : {Index(9), Index(12), Index(16), cloud.size()}) {
                const Index take = std::min(count, cloud.size());
                std::vector<Index> sub(cloud.begin(), cloud.begin() + static_cast<SignedIndex>(take));
                if (try_build_p2(sub, 500.0, ed.preferred)) {
                    ok_pref = true;
                    break;
                }
            }
            if (!ok_pref) {
                throw std::runtime_error("UnstructuredAdvectionSolver ENO3: degenerate cell stencil");
            }

            // Build 6 overlapping directional sector stencils around cell c
            for (int s = 0; s < 6; ++s) {
                const Real theta = static_cast<Real>(s) * (PI / 3.0);
                const Vec2 dir{std::cos(theta), std::sin(theta)};
                std::vector<std::pair<Real, Index>> ranked;
                ranked.reserve(cloud.size());
                for (Index nb : cloud) {
                    const Vec2 d = cells[nb].centroid - xc;
                    const Real len = d.norm();
                    if (len <= 1.0e-14) continue;
                    const Real cos_a = dot(d, dir) / len;
                    if (cos_a >= -0.2) {
                        ranked.emplace_back(len * (1.6 - cos_a), nb);
                    }
                }
                std::sort(ranked.begin(), ranked.end(),
                          [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
                for (Index take : {Index(6), Index(7), Index(8)}) {
                    if (ranked.size() < take) continue;
                    std::vector<Index> sub;
                    sub.reserve(take);
                    for (Index k = 0; k < take; ++k) {
                        sub.push_back(ranked[k].second);
                    }
                    P2Stencil sec;
                    if (try_build_p2(sub, 150.0, sec)) {
                        ed.sectors.push_back(std::move(sec));
                        break;
                    }
                }
            }
        }
    }

#ifdef FVM_USE_MPI
    if (use_mpi) {
        std::vector<Index> needed_u;
        std::vector<Index> needed_poly;
        for (Index c : active_cells_) {
            for (SignedIndex nb : cells[c].neighbor_ids) {
                if (nb >= 0) {
                    const Index nu = static_cast<Index>(nb);
                    if (decomp_->owner[nu] != my_rank) {
                        needed_u.push_back(nu);
                        if (scheme_ != ReconstructionScheme::FirstOrder) {
                            needed_poly.push_back(nu);
                        }
                    }
                }
            }
            if (need_linear_stencils) {
                for (Index nb : stencils_[c].neighbors) {
                    if (decomp_->owner[nb] != my_rank) needed_u.push_back(nb);
                }
            } else if ((scheme_ == ReconstructionScheme::ENO && eno_order_ == 3) ||
                       scheme_ == ReconstructionScheme::WENO) {
                for (Index nb : eno3_data_[c].preferred.neighbors) {
                    if (decomp_->owner[nb] != my_rank) needed_u.push_back(nb);
                }
                for (const auto& sec : eno3_data_[c].sectors) {
                    for (Index nb : sec.neighbors) {
                        if (decomp_->owner[nb] != my_rank) needed_u.push_back(nb);
                    }
                }
            }
        }
        u_sched_ = mpi::UnstructuredCommSchedule::build(
            decomp_->owner, my_rank, decomp_->size, decomp_->comm, std::move(needed_u));
        if (scheme_ != ReconstructionScheme::FirstOrder) {
            poly_sched_ = mpi::UnstructuredCommSchedule::build(
                decomp_->owner, my_rank, decomp_->size, decomp_->comm, std::move(needed_poly));
        }
    }
#endif

    cell_velocity_.resize(cells.size());
    std::vector<Real> div_accum(cells.size(), 0.0);
    std::vector<Real> out_flux_cap(cells.size(), 0.0);

    for (Index f = 0; f < faces.size(); ++f) {
        const auto& face = faces[f];
        const Real q = face_flux_rate_[f];

        div_accum[face.left_cell] += q;
        if (q > 0.0) {
            out_flux_cap[face.left_cell] += q;
        }

        if (face.right_cell >= 0) {
            const Index rc = static_cast<Index>(face.right_cell);
            div_accum[rc] -= q;
            if (q < 0.0) {
                out_flux_cap[rc] += -q;
            }
        }
    }

    max_discrete_div_ = 0.0;
    dt_cfl_limit_ = std::numeric_limits<Real>::infinity();
    for (Index c = 0; c < cells.size(); ++c) {
        cell_velocity_[c] = velocity_(cells[c].centroid.x, cells[c].centroid.y);
        const Real div_c = std::abs(div_accum[c]) / cells[c].area;
        max_discrete_div_ = std::max(max_discrete_div_, div_c);
        if (out_flux_cap[c] > 1.0e-30) {
            dt_cfl_limit_ = std::min(dt_cfl_limit_, cells[c].area / out_flux_cap[c]);
        }
    }
    if (!std::isfinite(dt_cfl_limit_)) {
        dt_cfl_limit_ = 1.0e-3;
    }
}

void UnstructuredAdvectionSolver::compute_limited_gradients(const std::vector<Real>& u,
                                                            std::vector<CellPoly2>& polys) const {
    const Index n_cells = mesh_.num_cells();
    const auto& cells = mesh_.cells();
    const auto& verts = mesh_.vertices();
    polys.assign(n_cells, CellPoly2{});

    Real u_glob_min = std::numeric_limits<Real>::infinity();
    Real u_glob_max = -std::numeric_limits<Real>::infinity();
    if (limiter_ == UnstructuredLimiter::Venkatakrishnan) {
        for (Index c : active_cells_) {
            u_glob_min = std::min(u_glob_min, u[c]);
            u_glob_max = std::max(u_glob_max, u[c]);
        }
        if (decomp_ && decomp_->size > 1) {
            u_glob_min = decomp_->allreduce_min(u_glob_min);
            u_glob_max = decomp_->allreduce_max(u_glob_max);
        }
    } else {
        u_glob_min = 0.0;
        u_glob_max = 0.0;
    }
    const Real u_scale = std::max(Real(1.0), u_glob_max - u_glob_min);
    constexpr Real kVenkatConst = 2.0;

    std::vector<Real> rhs;
    for (Index c : active_cells_) {
        const CellGradStencil& st = stencils_[c];
        const Index m = static_cast<Index>(st.neighbors.size());
        rhs.resize(m);

        const Real uc = u[c];
        Real u_min = uc;
        Real u_max = uc;
        for (Index r = 0; r < m; ++r) {
            const Real un = u[st.neighbors[r]];
            u_min = std::min(u_min, un);
            u_max = std::max(u_max, un);
            rhs[r] = (un - uc) * st.inv_distances[r];
        }

        const std::vector<Real> g = st.qr.solve(rhs);
        const Vec2 grad{g[0], g[1]};

        const Real dmax = u_max - uc;  // >= 0
        const Real dmin = u_min - uc;  // <= 0
        const Real h_c = std::sqrt(cells[c].area);
        const Real eps_sq = (limiter_ == UnstructuredLimiter::Venkatakrishnan)
            ? (u_scale * u_scale * std::pow(kVenkatConst * h_c, 3))
            : 0.0;

        Real phi_c = 1.0;
        const Vec2& xc = cells[c].centroid;
        for (Index vid : cells[c].vertex_ids) {
            const Real du = dot(grad, verts[vid] - xc);
            if (std::abs(du) <= 1.0e-14) {
                continue;
            }
            const Real delta_plus = (du > 0.0) ? dmax : dmin;
            Real phi_k = 1.0;
            if (limiter_ == UnstructuredLimiter::BarthJespersen) {
                const Real a = std::abs(delta_plus);
                const Real b = std::abs(du);
                if (a < b) {
                    phi_k = a / b;
                }
            } else {
                const Real a = std::abs(delta_plus);
                const Real b = std::abs(du);
                const Real num = a * a + 2.0 * a * b + eps_sq;
                const Real den = a * a + 2.0 * b * b + a * b + eps_sq;
                phi_k = (den > 0.0) ? std::min(Real(1.0), num / den) : Real(1.0);
            }
            phi_c = std::min(phi_c, phi_k);
        }
        polys[c].gx = phi_c * grad.x;
        polys[c].gy = phi_c * grad.y;
    }
}

void UnstructuredAdvectionSolver::compute_eno2_polys(const std::vector<Real>& u,
                                                     std::vector<CellPoly2>& polys) const {
    const Index n_cells = mesh_.num_cells();
    polys.assign(n_cells, CellPoly2{});
    std::vector<Real> rhs;
    for (Index c : active_cells_) {
        const CellGradStencil& st = stencils_[c];
        const Index m = static_cast<Index>(st.neighbors.size());
        rhs.resize(m);
        const Real uc = u[c];
        for (Index r = 0; r < m; ++r) {
            rhs[r] = (u[st.neighbors[r]] - uc) * st.inv_distances[r];
        }
        const std::vector<Real> g_pref = st.qr.solve(rhs);
        Real best_gx = g_pref[0];
        Real best_gy = g_pref[1];
        const Real beta_pref = best_gx * best_gx + best_gy * best_gy;

        Real min_sec_beta = std::numeric_limits<Real>::infinity();
        Real sec_gx = best_gx;
        Real sec_gy = best_gy;
        for (const auto& sp : eno2_pairs_[c]) {
            const Real b0 = u[sp.n0] - uc;
            const Real b1 = u[sp.n1] - uc;
            const Real gx = sp.inv00 * b0 + sp.inv01 * b1;
            const Real gy = sp.inv10 * b0 + sp.inv11 * b1;
            const Real beta = gx * gx + gy * gy;
            if (beta < min_sec_beta) {
                min_sec_beta = beta;
                sec_gx = gx;
                sec_gy = gy;
            }
        }
        if (4.0 * min_sec_beta < beta_pref) {
            best_gx = sec_gx;
            best_gy = sec_gy;
        }
        polys[c].gx = best_gx;
        polys[c].gy = best_gy;
    }
}

void UnstructuredAdvectionSolver::compute_p2_polys(const std::vector<Real>& u,
                                                   bool use_weno,
                                                   std::vector<CellPoly2>& polys) const {
    const Index n_cells = mesh_.num_cells();
    polys.assign(n_cells, CellPoly2{});
    std::vector<Real> rhs;

    auto solve_p2 = [&](const P2Stencil& st, Real uc, std::vector<Real>& coeff) {
        const Index m = static_cast<Index>(st.neighbors.size());
        rhs.resize(m);
        for (Index r = 0; r < m; ++r) {
            rhs[r] = st.weights[r] * (u[st.neighbors[r]] - uc);
        }
        coeff = st.qr.solve(rhs);
    };

    auto smoothness_p2 = [](const std::vector<Real>& a, const CellENO3Data& ed) -> Real {
        const Real t_m20 = (4.0 * a[2] * a[2] + a[3] * a[3]) * ed.m20_hat;
        const Real t_m11 = 4.0 * a[3] * (a[2] + a[4]) * ed.m11_hat;
        const Real t_m02 = (a[3] * a[3] + 4.0 * a[4] * a[4]) * ed.m02_hat;
        const Real h1_sq = a[0] * a[0] + a[1] * a[1] + t_m20 + t_m11 + t_m02;
        const Real h2_sq = 4.0 * a[2] * a[2] + 2.0 * a[3] * a[3] + 4.0 * a[4] * a[4];
        return std::max(Real(0.0), h1_sq) + h2_sq;
    };

    std::vector<Real> a_pref, a_sec, a_best_sec;
    std::vector<std::vector<Real>> sec_coeffs;
    std::vector<Real> sec_betas;
    for (Index c : active_cells_) {
        const CellENO3Data& ed = eno3_data_[c];
        const Real uc = u[c];
        solve_p2(ed.preferred, uc, a_pref);
        const Real beta_pref = smoothness_p2(a_pref, ed);

        Real chosen[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        if (!use_weno) {
            Real min_sec_beta = std::numeric_limits<Real>::infinity();
            for (const auto& sec : ed.sectors) {
                solve_p2(sec, uc, a_sec);
                const Real beta = smoothness_p2(a_sec, ed);
                if (beta < min_sec_beta) {
                    min_sec_beta = beta;
                    a_best_sec = a_sec;
                }
            }
            const std::vector<Real>& sel = (4.0 * min_sec_beta < beta_pref) ? a_best_sec : a_pref;
            for (int k = 0; k < 5; ++k) chosen[k] = sel[k];
        } else {
            const Index ns = ed.sectors.size();
            sec_coeffs.resize(ns);
            sec_betas.resize(ns);
            Real max_diff = 0.0;
            for (Index s = 0; s < ns; ++s) {
                solve_p2(ed.sectors[s], uc, sec_coeffs[s]);
                sec_betas[s] = smoothness_p2(sec_coeffs[s], ed);
                max_diff = std::max(max_diff, std::abs(beta_pref - sec_betas[s]));
            }
            const Real h = 1.0 / ed.inv_h;
            const Real eps = (weno_type_ == WENOWeightType::WENO_Z)
                ? std::max(1.0e-30, h * h * h * h)
                : 1.0e-8;
            auto raw_weight = [&](Real lambda, Real beta) -> Real {
                if (weno_type_ == WENOWeightType::WENO_Z) {
                    const Real r = max_diff / (beta + eps);
                    const Real r2 = r * r;
                    return lambda * (1.0 + r2 * r2);
                }
                const Real d = beta + eps;
                const Real d2 = d * d;
                return lambda / (d2 * d2);
            };
            constexpr Real kCentralLambda = 100.0;
            const Real w_pref = raw_weight(kCentralLambda, beta_pref);
            Real w_sum = w_pref;
            for (int k = 0; k < 5; ++k) chosen[k] = w_pref * a_pref[k];
            for (Index s = 0; s < ns; ++s) {
                const Real ws = raw_weight(1.0, sec_betas[s]);
                w_sum += ws;
                for (int k = 0; k < 5; ++k) chosen[k] += ws * sec_coeffs[s][k];
            }
            const Real inv_w = 1.0 / w_sum;
            for (int k = 0; k < 5; ++k) chosen[k] *= inv_w;
        }

        const Real inv_h = ed.inv_h;
        const Real inv_h2 = inv_h * inv_h;
        polys[c] = {
            chosen[0] * inv_h,
            chosen[1] * inv_h,
            chosen[2] * inv_h2,
            chosen[3] * inv_h2,
            chosen[4] * inv_h2,
            ed.m20, ed.m11, ed.m02
        };
    }
}

void UnstructuredAdvectionSolver::compute_rhs(const std::vector<Real>& u,
                                              Real t,
                                              std::vector<Real>& dudt,
                                              Real* boundary_flux_out,
                                              Real* outlet_flux_out) const {
    const Index n_cells = mesh_.num_cells();
    if (u.size() != n_cells) {
        throw std::invalid_argument("UnstructuredAdvectionSolver::compute_rhs state size mismatch");
    }
    dudt.assign(n_cells, 0.0);

    const bool use_mpi = (decomp_ != nullptr && decomp_->size > 1);
    const int my_rank = use_mpi ? decomp_->rank : 0;

    const std::vector<Real>* u_ptr = &u;
#ifdef FVM_USE_MPI
    if (use_mpi) {
        u_work_.resize(n_cells);
        for (Index c : active_cells_) {
            u_work_[c] = u[c];
        }
        u_sched_.exchange<Real>(
            decomp_->comm, 30,
            [&](Index c, Real* out) { out[0] = u_work_[c]; },
            [&](Index c, const Real* in) { u_work_[c] = in[0]; });
        u_ptr = &u_work_;
    }
#endif
    const std::vector<Real>& u_src = *u_ptr;

    const auto& faces = mesh_.faces();
    const auto& cells = mesh_.cells();
    const auto& areas = mesh_.cell_areas();
    const Index nq = static_cast<Index>(edge_quad_pts_);
    const bool use_poly = (scheme_ != ReconstructionScheme::FirstOrder);

    std::vector<CellPoly2> polys;
    if (scheme_ == ReconstructionScheme::TVD) {
        compute_limited_gradients(u_src, polys);
    } else if (scheme_ == ReconstructionScheme::ENO) {
        if (eno_order_ == 2) {
            compute_eno2_polys(u_src, polys);
        } else {
            compute_p2_polys(u_src, false, polys);
        }
    } else if (scheme_ == ReconstructionScheme::WENO) {
        compute_p2_polys(u_src, true, polys);
    }

#ifdef FVM_USE_MPI
    if (use_mpi && use_poly) {
        poly_sched_.exchange<CellPoly2>(
            decomp_->comm, 31,
            [&](Index c, Real* out) {
                const CellPoly2& p = polys[c];
                out[0] = p.gx; out[1] = p.gy;
                out[2] = p.qxx; out[3] = p.qxy; out[4] = p.qyy;
                out[5] = p.m20; out[6] = p.m11; out[7] = p.m02;
            },
            [&](Index c, const Real* in) {
                polys[c] = {in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]};
            });
    }
#endif

    Real bnd_flux = 0.0;
    Real outlet_flux = 0.0;

    for (Index f : active_faces_) {
        const auto& face = faces[f];
        const Real q = face_flux_rate_[f];
        const Real uL = u_src[face.left_cell];

        if (face.right_cell >= 0) {
            const Index rc = static_cast<Index>(face.right_cell);
            Real flux_val = 0.0;
            if (use_poly) {
                const Vec2& xcL = cells[face.left_cell].centroid;
                const Vec2& xcR = cells[rc].centroid;
                const Index q_base = f * nq;
                for (Index qi = 0; qi < nq; ++qi) {
                    const FaceQuadData& fq = face_quads_[q_base + qi];
                    const Vec2 xq{fq.x, fq.y};
                    const Real u_upw = (fq.weighted_flux >= 0.0)
                        ? (uL + polys[face.left_cell].eval_offset(xq - xcL))
                        : (u_src[rc] + polys[rc].eval_offset(xq - xcR));
                    flux_val += fq.weighted_flux * u_upw;
                }
            } else {
                const Real uR = u_src[rc];
                flux_val = (q >= 0.0) ? (q * uL) : (q * uR);
            }
            if (!use_mpi || decomp_->owner[face.left_cell] == my_rank) {
                dudt[face.left_cell] -= flux_val / areas[face.left_cell];
            }
            if (!use_mpi || decomp_->owner[rc] == my_rank) {
                dudt[rc] += flux_val / areas[rc];
            }
        } else {
            Real flux_val = 0.0;
            if (use_poly) {
                const Vec2& xcL = cells[face.left_cell].centroid;
                const Index q_base = f * nq;
                for (Index qi = 0; qi < nq; ++qi) {
                    const FaceQuadData& fq = face_quads_[q_base + qi];
                    const Real u_upw = (fq.weighted_flux >= 0.0)
                        ? (uL + polys[face.left_cell].eval_offset(Vec2{fq.x, fq.y} - xcL))
                        : inflow_bc_(fq.x, fq.y, t, face.boundary_tag);
                    flux_val += fq.weighted_flux * u_upw;
                }
            } else if (q >= 0.0) {
                flux_val = q * uL;
            } else {
                const Index q_base = f * nq;
                for (Index qi = 0; qi < nq; ++qi) {
                    const FaceQuadData& fq = face_quads_[q_base + qi];
                    const Real ug = (fq.weighted_flux < 0.0)
                        ? inflow_bc_(fq.x, fq.y, t, face.boundary_tag)
                        : uL;
                    flux_val += fq.weighted_flux * ug;
                }
            }

            if (q >= 0.0 && (face.boundary_tag == 2 || q > 0.0)) {
                outlet_flux += flux_val;
            }
            bnd_flux += flux_val;
            dudt[face.left_cell] -= flux_val / areas[face.left_cell];
        }
    }

    if (use_mpi) {
        bnd_flux = decomp_->allreduce_sum(bnd_flux);
        outlet_flux = decomp_->allreduce_sum(outlet_flux);
    }

    if (boundary_flux_out) *boundary_flux_out = bnd_flux;
    if (outlet_flux_out) *outlet_flux_out = outlet_flux;
}

UnstructuredRunSummary UnstructuredAdvectionSolver::run(
    std::vector<Real>& u,
    const UnstructuredRunOptions& opts,
    ExactDiagFn exact_diag) const {
    const auto t_start = std::chrono::steady_clock::now();

    const bool use_mpi = (decomp_ != nullptr && decomp_->size > 1);
    const int my_rank = use_mpi ? decomp_->rank : 0;
    const auto& areas = mesh_.cell_areas();

    auto owned_stats = [&](const std::vector<Real>& field) -> FieldStats {
        if (!use_mpi) {
            return compute_field_stats(field, areas);
        }
        FieldStats st;
        st.min_val = std::numeric_limits<Real>::infinity();
        st.max_val = -std::numeric_limits<Real>::infinity();
        Real mass = 0.0;
        for (Index c : active_cells_) {
            st.min_val = std::min(st.min_val, field[c]);
            st.max_val = std::max(st.max_val, field[c]);
            mass += field[c] * areas[c];
        }
        st.min_val = decomp_->allreduce_min(st.min_val);
        st.max_val = decomp_->allreduce_max(st.max_val);
        st.total_mass = decomp_->allreduce_sum(mass);
        return st;
    };

    UnstructuredRunSummary summary;
    summary.initial_stats = owned_stats(u);
    summary.max_discrete_divergence = max_discrete_div_;

    const bool want_vtk = !opts.vtk_dir.empty();
    io::PvdWriter pvd(opts.vtk_dir + "/" + opts.pvd_name);
    Index snap_idx = 0;

    std::vector<Real> u_exact, u_err, u_snap;
    auto write_snapshot = [&](Real t_now) {
        if (!want_vtk) return;
        std::ostringstream fname;
        fname << "step_" << std::setw(4) << std::setfill('0') << snap_idx++ << ".vtu";
        const std::string full_path = opts.vtk_dir + "/" + fname.str();
        const std::vector<Real>* u_out_ptr = &u;
        if (use_mpi) {
            u_snap = u;
            decomp_->allgather_owned(u_snap);
            u_out_ptr = &u_snap;
        }
        if (my_rank == 0) {
            if (exact_diag) {
                exact_diag(t_now, mesh_, *u_out_ptr, u_exact, u_err);
                io::write_vtu(full_path, mesh_,
                              {{"u", u_out_ptr}, {"u_exact", &u_exact}, {"error", &u_err}},
                              {{"velocity", &cell_velocity_}});
            } else {
                io::write_vtu(full_path, mesh_, {{"u", u_out_ptr}}, {{"velocity", &cell_velocity_}});
            }
            pvd.add_step(t_now, fname.str());
        }
    };

    Real t = 0.0;
    Real next_output_t = (opts.output_interval > 0.0) ? opts.output_interval : opts.t_end + 1.0;
    write_snapshot(t);

    const Real dt_nom = opts.cfl * dt_cfl_limit_;
    summary.dt_used = dt_nom;

    const std::vector<Real> stage_weights = time::rk_stage_weights(opts.rk_method);
    Index stage_idx = 0;
    Real step_bnd_flux = 0.0;
    Real step_outlet_flux = 0.0;

    time::RKStepper<std::vector<Real>> stepper(opts.rk_method);
    auto rhs_fn = [&](const std::vector<Real>& state, Real stage_t, std::vector<Real>& dudt) {
        Real bnd_flux = 0.0;
        Real out_flux = 0.0;
        this->compute_rhs(state, stage_t, dudt, &bnd_flux, &out_flux);
        const Real w = stage_weights.at(stage_idx++);
        step_bnd_flux += w * bnd_flux;
        step_outlet_flux += w * out_flux;
    };

    auto compute_mass = [&](const std::vector<Real>& field) {
        Real m = 0.0;
        for (Index c : active_cells_) {
            m += areas[c] * field[c];
        }
        return use_mpi ? decomp_->allreduce_sum(m) : m;
    };
    Real mass_prev = summary.initial_stats.total_mass;

    while (t < opts.t_end - 1.0e-13) {
        Real dt = dt_nom;
        if (t + dt > opts.t_end) {
            dt = opts.t_end - t;
        }
        stage_idx = 0;
        step_bnd_flux = 0.0;
        step_outlet_flux = 0.0;
        stepper.step(u, t, dt, rhs_fn);
        t += dt;
        summary.steps += 1;

        const Real mass_now = compute_mass(u);
        const Real step_res = std::abs((mass_now - mass_prev) + dt * step_bnd_flux);
        summary.max_conservation_residual = std::max(summary.max_conservation_residual, step_res);
        summary.net_boundary_outflow += dt * step_bnd_flux;
        summary.outlet_outflow_mass += dt * step_outlet_flux;
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

    if (use_mpi) {
        decomp_->allgather_owned(u);
    }

    const auto t_stop = std::chrono::steady_clock::now();
    summary.final_time = t;
    summary.runtime_sec = std::chrono::duration<Real>(t_stop - t_start).count();
    summary.final_stats = owned_stats(u);
    summary.cumulative_conservation_residual = std::abs(
        (summary.final_stats.total_mass - summary.initial_stats.total_mass) +
        summary.net_boundary_outflow);
    return summary;
}

}  // namespace fvm::solvers
