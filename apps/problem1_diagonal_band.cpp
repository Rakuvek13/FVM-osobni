#include "fvm/core/convergence.hpp"
#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/geometry_exact.hpp"
#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/solvers/structured_solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace fvm;

std::vector<Index> parse_grids(const std::string& s) {
    std::vector<Index> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(static_cast<Index>(std::stoul(item)));
        }
    }
    return out;
}

struct NormalProfileMetrics {
    Real peak_mid = 0.0;
    Real peak_loss = 0.0;
    Real thickness_10_90_mid = -1.0;    // along x + y = 1.0
    Real thickness_10_90_early = -1.0;  // along x + y = 0.5
};

Real measure_10_90_along_antidiag(const StructuredMesh2D& mesh,
                                  const StructuredField2D& u,
                                  Index sum_ij,
                                  Real w,
                                  Real& peak_out) {
    const Index N = mesh.nx();
    std::vector<Real> eta;
    std::vector<Real> val;

    const Index i_max = std::min(sum_ij, N - 1);
    const Index i_min = (sum_ij >= N - 1) ? (sum_ij - (N - 1)) : 0;
    peak_out = 0.0;
    for (Index ii = i_max + 1; ii > i_min; --ii) {
        const Index i = ii - 1;
        const Index j = sum_ij - i;
        const Vec2 xc = mesh.cell_center(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j));
        const Real e = (xc.y - xc.x) / std::sqrt(2.0);
        const Real v = u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j));
        eta.push_back(e);
        val.push_back(v);
        if (std::abs(e) < 0.5 * w / std::sqrt(2.0)) {
            peak_out = std::max(peak_out, v);
        }
    }

    Real eta_90 = -1.0, eta_10 = -1.0;
    bool found_90 = false, found_10 = false;
    for (Index k = 0; k + 1 < eta.size(); ++k) {
        if (eta[k] < 0.0 && eta[k + 1] < 0.0) continue;
        const Real u0 = val[k], u1 = val[k + 1];
        if (!found_90 && u0 >= 0.9 && u1 < 0.9) {
            eta_90 = eta[k] + ((0.9 - u0) / (u1 - u0)) * (eta[k + 1] - eta[k]);
            found_90 = true;
        }
        if (!found_10 && u0 >= 0.1 && u1 < 0.1) {
            eta_10 = eta[k] + ((0.1 - u0) / (u1 - u0)) * (eta[k + 1] - eta[k]);
            found_10 = true;
        }
    }
    return (found_90 && found_10 && eta_10 >= eta_90) ? (eta_10 - eta_90) : -1.0;
}

NormalProfileMetrics analyze_normal_profile(const StructuredMesh2D& mesh,
                                            const StructuredField2D& u,
                                            const StructuredField2D& u_exact,
                                            Real w,
                                            const std::string& csv_path) {
    const Index N = mesh.nx();
    NormalProfileMetrics m;
    Real peak_early = 0.0;
    m.thickness_10_90_early = measure_10_90_along_antidiag(mesh, u, N / 2 - 1, w, peak_early);
    m.thickness_10_90_mid = measure_10_90_along_antidiag(mesh, u, N - 1, w, m.peak_mid);

    if (!csv_path.empty()) {
        std::filesystem::create_directories(std::filesystem::path(csv_path).parent_path());
        std::ofstream out(csv_path);
        out << std::setprecision(15) << "eta,u,u_exact\n";
        for (Index k = 0; k < N; ++k) {
            const SignedIndex i = static_cast<SignedIndex>(N - 1 - k);
            const SignedIndex j = static_cast<SignedIndex>(k);
            const Vec2 xc = mesh.cell_center(i, j);
            const Real e = (xc.y - xc.x) / std::sqrt(2.0);
            out << e << ',' << u(i, j) << ',' << u_exact(i, j) << '\n';
        }
    }

    m.peak_loss = 1.0 - m.peak_mid;
    return m;
}

void init_diagonal_band(const StructuredMesh2D& mesh, Real w, StructuredField2D& f) {
    const Real dx = mesh.dx();
    const Real dy = mesh.dy();
    const Real area = mesh.cell_area();
    for (Index j = 0; j < mesh.ny(); ++j) {
        const Real y0 = mesh.ymin() + static_cast<Real>(j) * dy;
        const Real y1 = y0 + dy;
        for (Index i = 0; i < mesh.nx(); ++i) {
            const Real x0 = mesh.xmin() + static_cast<Real>(i) * dx;
            const Real x1 = x0 + dx;
            const std::vector<Vec2> poly = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
            f(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) =
                geom::intersection_area_diagonal_band(poly, w) / area;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    mpi::MpiGuard mpi_guard(argc, argv);
    const bool is_root = (mpi_guard.rank() == 0);

    std::string scheme_arg = "all";
    Real t_end = 1.0;
    Real cfl = 0.45;
    Real w = 0.2;
    std::vector<Index> grids = {40, 80, 160, 320};
    std::string out_dir = "output/problem1";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scheme" && i + 1 < argc) scheme_arg = argv[++i];
        else if (arg == "--t" && i + 1 < argc) t_end = std::stod(argv[++i]);
        else if (arg == "--cfl" && i + 1 < argc) cfl = std::stod(argv[++i]);
        else if (arg == "--w" && i + 1 < argc) w = std::stod(argv[++i]);
        else if (arg == "--grids" && i + 1 < argc) grids = parse_grids(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    }

    const auto schemes = solvers::parse_scheme_selection(scheme_arg);

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletFaceAvg;
    bc.left.face_avg = [w](Real y0, Real y1, Real) {
        return geom::interval_overlap_fraction(y0, y1, 0.0, w);
    };
    bc.bottom.type = StructuredBCType::DirichletFaceAvg;
    bc.bottom.face_avg = [w](Real x0, Real x1, Real) {
        return geom::interval_overlap_fraction(x0, x1, 0.0, w);
    };
    bc.right.type = StructuredBCType::ZeroGradient;
    bc.top.type = StructuredBCType::ZeroGradient;

    const auto law = solvers::make_linear_advection_law(1.0, 1.0);
    std::vector<ConvergenceRow> rows;
    if (is_root) {
        std::filesystem::create_directories(out_dir);
        std::cout << "Problem 1: Diagonal Inviscid Band on (0,1)^2 (a=(1,1), w=" << w
                  << ", T=" << t_end << ", CFL=" << cfl
                  << ", ranks=" << mpi_guard.size() << ")\n";
    }

    for (const auto scheme : schemes) {
        std::string scheme_label = "1st-Order";
        std::string scheme_tag = "first";
        time::RKMethod rk = time::RKMethod::Euler;
        if (scheme == solvers::ReconstructionScheme::TVD) {
            scheme_label = "TVD-mc";
            scheme_tag = "tvd";
            rk = time::RKMethod::SSP_RK2;
        } else if (scheme == solvers::ReconstructionScheme::ENO) {
            scheme_label = "ENO3";
            scheme_tag = "eno3";
            rk = time::RKMethod::SSP_RK3;
        } else if (scheme == solvers::ReconstructionScheme::WENO) {
            scheme_label = "WENO5-z";
            scheme_tag = "weno5";
            rk = time::RKMethod::SSP_RK3;
        }
        if (is_root) {
            std::cout << "\n[" << scheme_label << "]\n";
        }

        for (Index idx = 0; idx < grids.size(); ++idx) {
            const Index n = grids[idx];
            const auto decomp = mpi::CartDecomp2D::create(n, n, false, false);
            StructuredMesh2D mesh = decomp.make_local_mesh(0.0, 1.0, 0.0, 1.0, 3);
            auto u = mesh.create_field();
            auto u_exact = mesh.create_field();
            init_diagonal_band(mesh, w, u);
            init_diagonal_band(mesh, w, u_exact);

            solvers::StructuredFVSolver solver(
                mesh, bc, law, scheme,
                solvers::TVDSlopeLimiter::MC, 3, solvers::WENOWeightType::WENO_Z, &decomp);

            solvers::SolverRunOptions opts;
            opts.scheme = scheme;
            opts.tvd_limiter = solvers::TVDSlopeLimiter::MC;
            opts.eno_order = 3;
            opts.weno_type = solvers::WENOWeightType::WENO_Z;
            opts.rk_method = rk;
            opts.cfl = cfl;
            opts.t_end = t_end;
            opts.output_interval = t_end / 10.0;
            if (idx + 1 == grids.size()) {
                opts.vtk_dir = out_dir + "/" + scheme_tag + "_N" + std::to_string(n);
            }

            auto exact_diag = [w](Real, const StructuredMesh2D& m, const StructuredField2D& num,
                                  StructuredField2D& exact_out, StructuredField2D& err_out) {
                init_diagonal_band(m, w, exact_out);
                for (Index j = 0; j < m.ny(); ++j) {
                    for (Index i = 0; i < m.nx(); ++i) {
                        const SignedIndex is = static_cast<SignedIndex>(i);
                        const SignedIndex js = static_cast<SignedIndex>(j);
                        err_out(is, js) = std::abs(num(is, js) - exact_out(is, js));
                    }
                }
            };

            const auto summary = solver.run(u, opts, exact_diag);
            const auto norms = mpi::compute_error_norms(u, u_exact, mesh.cell_area(), decomp);
            const auto u_global = mpi::gather_structured_field(u, decomp);

            if (is_root) {
                StructuredMesh2D global_mesh(0.0, 1.0, 0.0, 1.0, n, n, 0);
                auto u_exact_global = global_mesh.create_field();
                init_diagonal_band(global_mesh, w, u_exact_global);
                const auto prof = analyze_normal_profile(
                    global_mesh, u_global, u_exact_global, w,
                    out_dir + "/profile_" + scheme_tag + "_N" + std::to_string(n) + ".csv");

                auto fmt_thick = [](Real d) {
                    if (d < 0.0) return std::string("merged(<0.9)");
                    std::ostringstream oss;
                    oss << std::setprecision(5) << d;
                    return oss.str();
                };

                std::cout << "  N=" << n << "x" << n
                          << " | steps=" << summary.steps
                          << " | u in [" << summary.final_stats.min_val << ", " << summary.final_stats.max_val << "]"
                          << " | peak_mid=" << prof.peak_mid << " (loss=" << prof.peak_loss << ")"
                          << " | d10_90(x+y=0.5)=" << fmt_thick(prof.thickness_10_90_early)
                          << " | d10_90(x+y=1.0)=" << fmt_thick(prof.thickness_10_90_mid)
                          << " | max_RM=" << summary.max_conservation_residual << "\n";

                rows.push_back({
                    scheme_label,
                    global_mesh.num_cells(),
                    global_mesh.dx(),
                    norms,
                    0.0, 0.0, 0.0,
                    summary.max_conservation_residual,
                    summary.runtime_sec
                });
            }
        }
    }

    if (is_root) {
        populate_eoc(rows);
        std::cout << "\n";
        print_convergence_table(std::cout, rows);
        write_convergence_csv(out_dir + "/convergence.csv", rows);
        std::cout << "Saved ParaView output and normal profiles to " << out_dir << "/\n";
    }
    return 0;
}
