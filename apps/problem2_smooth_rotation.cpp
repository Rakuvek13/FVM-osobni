#include "fvm/core/convergence.hpp"
#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/solvers/structured_solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct RotationMetrics {
    Real peak_abs = 0.0;
    Real exact_peak_abs = 0.0;
    Real peak_retention = 0.0;
    Real l2_norm = 0.0;
    Real exact_l2_norm = 0.0;
    Real l2_retention = 0.0;
};

RotationMetrics analyze_rotation_field(const StructuredMesh2D& mesh,
                                       const StructuredField2D& u,
                                       const StructuredField2D& u_exact,
                                       Real y_slice,
                                       const std::string& csv_path) {
    RotationMetrics m;
    Real sum_u2 = 0.0;
    Real sum_ex2 = 0.0;
    const Index nx = mesh.nx();
    const Index ny = mesh.ny();
    for (Index j = 0; j < ny; ++j) {
        const SignedIndex js = static_cast<SignedIndex>(j);
        for (Index i = 0; i < nx; ++i) {
            const SignedIndex is = static_cast<SignedIndex>(i);
            const Real val = u(is, js);
            const Real ex = u_exact(is, js);
            m.peak_abs = std::max(m.peak_abs, std::abs(val));
            m.exact_peak_abs = std::max(m.exact_peak_abs, std::abs(ex));
            sum_u2 += val * val;
            sum_ex2 += ex * ex;
        }
    }
    const Real area = mesh.cell_area();
    m.l2_norm = std::sqrt(sum_u2 * area);
    m.exact_l2_norm = std::sqrt(sum_ex2 * area);
    m.peak_retention = (m.exact_peak_abs > 0.0) ? (m.peak_abs / m.exact_peak_abs) : 0.0;
    m.l2_retention = (m.exact_l2_norm > 0.0) ? (m.l2_norm / m.exact_l2_norm) : 0.0;

    if (!csv_path.empty()) {
        std::filesystem::create_directories(std::filesystem::path(csv_path).parent_path());
        std::ofstream out(csv_path);
        out << std::setprecision(15) << "x,y,u,u_exact\n";
        Index j_best = 0;
        Real best_dy = std::numeric_limits<Real>::infinity();
        for (Index j = 0; j < ny; ++j) {
            const Real y = mesh.cell_center(0, static_cast<SignedIndex>(j)).y;
            const Real d = std::abs(y - y_slice);
            if (d < best_dy) {
                best_dy = d;
                j_best = j;
            }
        }
        const SignedIndex js = static_cast<SignedIndex>(j_best);
        for (Index i = 0; i < nx; ++i) {
            const SignedIndex is = static_cast<SignedIndex>(i);
            const Vec2 xc = mesh.cell_center(is, js);
            out << xc.x << ',' << xc.y << ',' << u(is, js) << ',' << u_exact(is, js) << '\n';
        }
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    mpi::MpiGuard mpi_guard(argc, argv);
    const bool is_root = (mpi_guard.rank() == 0);

    std::string scheme_arg = "all";
    Real t_end = 1.0;
    Real cfl = 0.45;
    Real sigma = 0.06;
    Real k_wave = 6.0 * PI;
    std::vector<Index> grids = {40, 80, 160, 320};
    std::string out_dir = "output/problem2";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scheme" && i + 1 < argc) scheme_arg = argv[++i];
        else if (arg == "--t" && i + 1 < argc) t_end = std::stod(argv[++i]);
        else if (arg == "--cfl" && i + 1 < argc) cfl = std::stod(argv[++i]);
        else if (arg == "--sigma" && i + 1 < argc) sigma = std::stod(argv[++i]);
        else if (arg == "--k" && i + 1 < argc) k_wave = std::stod(argv[++i]);
        else if (arg == "--grids" && i + 1 < argc) grids = parse_grids(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    }

    const auto specs = solvers::parse_scheme_specs(scheme_arg);

    const Real x0 = 0.5;
    const Real y0 = 0.70;
    const Real inv_2sig2 = 1.0 / (2.0 * sigma * sigma);
    auto u0_fn = [x0, y0, inv_2sig2, k_wave](Real x, Real y) -> Real {
        const Real dx = x - x0;
        const Real dy = y - y0;
        return std::exp(-(dx * dx + dy * dy) * inv_2sig2) * std::cos(k_wave * dx);
    };
    auto u_exact_fn = [u0_fn](Real x, Real y, Real t) -> Real {
        const Real theta = 2.0 * PI * t;
        const Real c = std::cos(theta);
        const Real s = std::sin(theta);
        const Real xr = x - 0.5;
        const Real yr = y - 0.5;
        const Real xp = 0.5 + xr * c + yr * s;
        const Real yp = 0.5 - xr * s + yr * c;
        return u0_fn(xp, yp);
    };

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletCellAvg;
    bc.left.value = u_exact_fn;
    bc.right = bc.left;
    bc.bottom = bc.left;
    bc.top = bc.left;

    solvers::StructuredConservationLaw law;
    law.fx = [](Real, Real y, Real u) { return -2.0 * PI * (y - 0.5) * u; };
    law.fy = [](Real x, Real, Real u) { return 2.0 * PI * (x - 0.5) * u; };
    law.alpha_x = [](Real, Real y, Real, Real) { return std::abs(-2.0 * PI * (y - 0.5)); };
    law.alpha_y = [](Real x, Real, Real, Real) { return std::abs(2.0 * PI * (x - 0.5)); };
    law.max_speeds = [](const StructuredField2D&, Real& max_ax, Real& max_ay) {
        max_ax = PI;
        max_ay = PI;
    };

    std::vector<ConvergenceRow> rows;
    if (is_root) {
        std::filesystem::create_directories(out_dir);
        std::cout << "Problem 2: Smooth Solid-Body Rotation of Modulated Wave Packet on (0,1)^2\n"
                  << "  v = (-2*pi*(y-0.5), 2*pi*(x-0.5)), center=(0.5,0.70), sigma=" << sigma
                  << ", k=" << k_wave / PI << "*pi, T=" << t_end << ", CFL=" << cfl
                  << ", ranks=" << mpi_guard.size() << "\n";
    }

    for (const auto spec : specs) {
        const auto scheme = spec.scheme;
        std::string scheme_label = "1st-Order";
        std::string scheme_tag = "first";
        time::RKMethod rk = time::RKMethod::Euler;
        int eno_order = 3;
        int weno_order = 5;
        if (scheme == solvers::ReconstructionScheme::TVD) {
            scheme_label = "TVD-mc";
            scheme_tag = "tvd";
            rk = time::RKMethod::SSP_RK2;
        } else if (scheme == solvers::ReconstructionScheme::ENO) {
            eno_order = spec.order;
            scheme_label = (eno_order == 2) ? "ENO2" : "ENO3";
            scheme_tag = (eno_order == 2) ? "eno2" : "eno3";
            rk = (eno_order == 2) ? time::RKMethod::SSP_RK2 : time::RKMethod::SSP_RK3;
        } else if (scheme == solvers::ReconstructionScheme::WENO) {
            weno_order = spec.order;
            scheme_label = (weno_order == 3) ? "WENO3-z" : "WENO5-z";
            scheme_tag = (weno_order == 3) ? "weno3" : "weno5";
            rk = (weno_order == 3) ? time::RKMethod::SSP_RK3 : time::RKMethod::RK4;
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
            mesh.initialize_cell_averages(u, u0_fn, 5);
            mesh.initialize_cell_averages(
                u_exact, [&](Real x, Real y) { return u_exact_fn(x, y, t_end); }, 5);

            solvers::StructuredFVSolver solver(
                mesh, bc, law, scheme,
                solvers::TVDSlopeLimiter::MC, eno_order, solvers::WENOWeightType::WENO_Z,
                &decomp, weno_order);

            solvers::SolverRunOptions opts;
            opts.scheme = scheme;
            opts.tvd_limiter = solvers::TVDSlopeLimiter::MC;
            opts.eno_order = eno_order;
            opts.weno_order = weno_order;
            opts.weno_type = solvers::WENOWeightType::WENO_Z;
            opts.rk_method = rk;
            opts.cfl = cfl;
            opts.t_end = t_end;
            opts.output_interval = t_end / 10.0;
            if (idx + 1 == grids.size()) {
                opts.vtk_dir = out_dir + "/" + scheme_tag + "_N" + std::to_string(n);
            }

            auto exact_diag = [&](Real t_now, const StructuredMesh2D& m, const StructuredField2D& num,
                                  StructuredField2D& ex_out, StructuredField2D& err_out) {
                m.initialize_cell_averages(
                    ex_out, [&](Real x, Real y) { return u_exact_fn(x, y, t_now); }, 5);
                for (Index j = 0; j < m.ny(); ++j) {
                    for (Index i = 0; i < m.nx(); ++i) {
                        const SignedIndex is = static_cast<SignedIndex>(i);
                        const SignedIndex js = static_cast<SignedIndex>(j);
                        err_out(is, js) = std::abs(num(is, js) - ex_out(is, js));
                    }
                }
            };

            const auto summary = solver.run(u, opts, exact_diag);
            const auto norms = mpi::compute_error_norms(u, u_exact, mesh.cell_area(), decomp);
            const auto u_global = mpi::gather_structured_field(u, decomp);
            const auto u_exact_global = mpi::gather_structured_field(u_exact, decomp);

            if (is_root) {
                StructuredMesh2D global_mesh(0.0, 1.0, 0.0, 1.0, n, n, 0);
                const auto metrics = analyze_rotation_field(
                    global_mesh, u_global, u_exact_global, y0,
                    out_dir + "/profile_" + scheme_tag + "_N" + std::to_string(n) + ".csv");

                std::cout << "  N=" << n << "x" << n
                          << " | steps=" << summary.steps
                          << " | u in [" << summary.final_stats.min_val << ", " << summary.final_stats.max_val << "]"
                          << " | R_peak=" << metrics.peak_retention
                          << " | R_L2=" << metrics.l2_retention
                          << " | E1=" << norms.l1
                          << " | Einf=" << norms.linf
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
        std::cout << "Saved ParaView output and profiles to " << out_dir << "/\n";
    }
    return 0;
}
