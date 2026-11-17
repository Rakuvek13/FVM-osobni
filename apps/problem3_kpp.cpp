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

StructuredField2D restrict_by_averaging(const StructuredField2D& fine, Index nx, Index ny) {
    const Index rx = fine.nx() / nx;
    const Index ry = fine.ny() / ny;
    StructuredField2D coarse(nx, ny, 0, 0.0);
    const Real inv_factor = 1.0 / static_cast<Real>(rx * ry);
    for (Index j = 0; j < ny; ++j) {
        for (Index i = 0; i < nx; ++i) {
            Real sum = 0.0;
            for (Index sj = 0; sj < ry; ++sj) {
                for (Index si = 0; si < rx; ++si) {
                    sum += fine(static_cast<SignedIndex>(i * rx + si),
                                static_cast<SignedIndex>(j * ry + sj));
                }
            }
            coarse(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = sum * inv_factor;
        }
    }
    return coarse;
}

void initialize_kpp_exact_fractions(const StructuredMesh2D& mesh,
                                    StructuredField2D& u,
                                    Real u_in,
                                    Real u_out) {
    const Real dx = mesh.dx();
    const Real dy = mesh.dy();
    const Real area = mesh.cell_area();
    for (Index j = 0; j < mesh.ny(); ++j) {
        const Real y0 = mesh.ymin() + static_cast<Real>(j) * dy;
        const Real y1 = y0 + dy;
        for (Index i = 0; i < mesh.nx(); ++i) {
            const Real x0 = mesh.xmin() + static_cast<Real>(i) * dx;
            const Real x1 = x0 + dx;
            const Real disk_area = geom::disk_intersection_area_rect(x0, x1, y0, y1, 1.0);
            const Real frac = disk_area / area;
            u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) =
                u_out + (u_in - u_out) * frac;
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
    Real domain_half = 2.5;  // Omega = (-2.5, 2.5)^2
    std::vector<Index> grids = {100, 200, 400, 800};
    Index ref_n = 1600;      // fine monotone reference so all 4 solvers compare on N=100..800
    std::string out_dir = "output/problem3";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scheme" && i + 1 < argc) scheme_arg = argv[++i];
        else if (arg == "--t" && i + 1 < argc) t_end = std::stod(argv[++i]);
        else if (arg == "--cfl" && i + 1 < argc) cfl = std::stod(argv[++i]);
        else if (arg == "--grids" && i + 1 < argc) grids = parse_grids(argv[++i]);
        else if (arg == "--ref" && i + 1 < argc) ref_n = static_cast<Index>(std::stoul(argv[++i]));
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    }

    const auto schemes = solvers::parse_scheme_selection(scheme_arg);
    const Real u_in = 3.5 * PI;
    const Real u_out = 0.25 * PI;

    solvers::StructuredConservationLaw law;
    law.fx = [](Real, Real, Real u) { return std::sin(u); };
    law.fy = [](Real, Real, Real u) { return std::cos(u); };
    law.alpha_x = [](Real, Real, Real, Real) { return 1.0; };
    law.alpha_y = [](Real, Real, Real, Real) { return 1.0; };
    law.max_speeds = [](const StructuredField2D&, Real& max_ax, Real& max_ay) {
        max_ax = 1.0;
        max_ay = 1.0;
    };

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletExact;
    bc.left.value = [u_out](Real, Real, Real) { return u_out; };
    bc.right = bc.left;
    bc.bottom = bc.left;
    bc.top = bc.left;

    StructuredField2D ref_field;
    bool has_ref = false;
    if (ref_n > 0) {
        if (is_root) {
            std::cout << "Computing KPP monotone reference on " << ref_n << "x" << ref_n
                      << " grid (ranks=" << mpi_guard.size() << ")...\n";
        }
        const auto ref_decomp = mpi::CartDecomp2D::create(ref_n, ref_n, false, false);
        StructuredMesh2D ref_mesh = ref_decomp.make_local_mesh(
            -domain_half, domain_half, -domain_half, domain_half, 1);
        auto local_ref = ref_mesh.create_field();
        initialize_kpp_exact_fractions(ref_mesh, local_ref, u_in, u_out);
        solvers::StructuredFVSolver ref_solver(
            ref_mesh, bc, law, solvers::ReconstructionScheme::FirstOrder,
            solvers::TVDSlopeLimiter::Minmod, 3, solvers::WENOWeightType::WENO_Z, &ref_decomp);
        solvers::SolverRunOptions ref_opts;
        ref_opts.rk_method = time::RKMethod::Euler;
        ref_opts.cfl = cfl;
        ref_opts.t_end = t_end;
        const auto ref_sum = ref_solver.run(local_ref, ref_opts);
        ref_field = mpi::gather_structured_field(local_ref, ref_decomp);
        has_ref = true;
        if (is_root) {
            std::cout << "  Reference done: steps=" << ref_sum.steps
                      << ", time=" << ref_sum.runtime_sec << "s"
                      << ", u in [" << ref_sum.final_stats.min_val << ", " << ref_sum.final_stats.max_val << "]"
                      << ", max_RM=" << ref_sum.max_conservation_residual << "\n";
        }
    }

    std::vector<ConvergenceRow> rows;
    if (is_root) {
        std::filesystem::create_directories(out_dir);
        std::cout << "Problem 3: KPP Rotating Wave (domain=[-2.5,2.5]^2, T=" << t_end
                  << ", CFL=" << cfl << ", alpha=1, u_out=pi/4, ranks=" << mpi_guard.size() << ")\n";
    }

    for (const auto scheme : schemes) {
        std::string scheme_label = "1st-Order";
        std::string scheme_tag = "first";
        time::RKMethod rk = time::RKMethod::Euler;
        if (scheme == solvers::ReconstructionScheme::TVD) {
            scheme_label = "TVD-minmod";
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
            StructuredMesh2D mesh = decomp.make_local_mesh(
                -domain_half, domain_half, -domain_half, domain_half, 3);
            auto u = mesh.create_field();
            initialize_kpp_exact_fractions(mesh, u, u_in, u_out);

            solvers::StructuredFVSolver solver(
                mesh, bc, law, scheme,
                solvers::TVDSlopeLimiter::Minmod, 3, solvers::WENOWeightType::WENO_Z, &decomp);
            solvers::SolverRunOptions opts;
            opts.scheme = scheme;
            opts.tvd_limiter = solvers::TVDSlopeLimiter::Minmod;
            opts.eno_order = 3;
            opts.weno_type = solvers::WENOWeightType::WENO_Z;
            opts.rk_method = rk;
            opts.cfl = cfl;
            opts.t_end = t_end;
            opts.output_interval = t_end / 10.0;
            if (idx + 1 == grids.size()) {
                opts.vtk_dir = out_dir + "/" + scheme_tag + "_N" + std::to_string(n);
            }

            const auto summary = solver.run(u, opts);

            ErrorNorms norms;
            const bool is_self_ref =
                (scheme == solvers::ReconstructionScheme::FirstOrder && has_ref && n == ref_n);
            const bool can_compare_ref = has_ref && !is_self_ref && ref_n % n == 0 && ref_n >= n;
            if (can_compare_ref) {
                StructuredField2D ref_coarse_glob;
                if (is_root) {
                    ref_coarse_glob = (ref_n == n) ? ref_field : restrict_by_averaging(ref_field, n, n);
                }
                auto local_ref_coarse = mesh.create_field();
                mpi::scatter_structured_field(ref_coarse_glob, local_ref_coarse, decomp);
                norms = mpi::compute_error_norms(u, local_ref_coarse, mesh.cell_area(), decomp);
            }

            if (is_root) {
                std::cout << "  N=" << n << "x" << n
                          << " | steps=" << summary.steps
                          << " | u in [" << summary.final_stats.min_val << ", " << summary.final_stats.max_val << "]";
                if (is_self_ref) {
                    std::cout << " | (reference grid)";
                } else if (can_compare_ref) {
                    std::cout << " | E1_vs_ref=" << norms.l1
                              << " | Einf_vs_ref=" << norms.linf;
                }
                std::cout << " | max_RM=" << summary.max_conservation_residual << "\n";

                if (can_compare_ref) {
                    rows.push_back({
                        scheme_label,
                        n * n,
                        mesh.dx(),
                        norms,
                        0.0, 0.0, 0.0,
                        summary.max_conservation_residual,
                        summary.runtime_sec
                    });
                }
            }
        }
    }

    if (is_root) {
        populate_eoc(rows);
        std::cout << "\n";
        print_convergence_table(std::cout, rows);
        write_convergence_csv(out_dir + "/convergence.csv", rows);
        std::cout << "Saved ParaView output and convergence table to " << out_dir << "/\n";
    }
    return 0;
}
