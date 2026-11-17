#include "fvm/core/convergence.hpp"
#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/geometry_exact.hpp"
#include "fvm/mesh/mesh_generator.hpp"
#include "fvm/mesh/unstructured_mesh.hpp"
#include "fvm/solvers/unstructured_solver.hpp"

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

std::vector<Index> parse_list(const std::string& s) {
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

struct TubeSmearingStats {
    Real mass_total = 0.0;
    Real cross_stream_leakage = 0.0;  // mass outside strip y in [0.25, 0.75]
    Real streamwise_spill = 0.0;      // mass inside strip y in [0.25, 0.75] outside x in [2.5, 3.0]
    Real sigma_x = 0.0;
    Real sigma_y = 0.0;
};

TubeSmearingStats compute_tube_smearing(const UnstructuredMesh2D& mesh,
                                        const std::vector<Real>& u,
                                        Real x_min_patch, Real x_max_patch,
                                        Real y_min_patch, Real y_max_patch) {
    TubeSmearingStats st;
    Real mx = 0.0, my = 0.0;
    const auto& cells = mesh.cells();
    for (Index c = 0; c < cells.size(); ++c) {
        const Real area = cells[c].area;
        const Real cell_mass = u[c] * area;
        st.mass_total += cell_mass;
        mx += cell_mass * cells[c].centroid.x;
        my += cell_mass * cells[c].centroid.y;

        const auto poly = mesh.cell_polygon(c);
        const Real strip_area = geom::intersection_area_box(poly, 0.0, 4.0, y_min_patch, y_max_patch);
        const Real patch_area = geom::intersection_area_box(
            poly, x_min_patch, x_max_patch, y_min_patch, y_max_patch);
        st.cross_stream_leakage += u[c] * std::max(0.0, area - strip_area);
        st.streamwise_spill += u[c] * std::max(0.0, strip_area - patch_area);
    }

    if (st.mass_total > 1.0e-30) {
        const Real mean_x = mx / st.mass_total;
        const Real mean_y = my / st.mass_total;
        Real var_x = 0.0, var_y = 0.0;
        for (Index c = 0; c < cells.size(); ++c) {
            const Real w = u[c] * cells[c].area / st.mass_total;
            const Real dx = cells[c].centroid.x - mean_x;
            const Real dy = cells[c].centroid.y - mean_y;
            var_x += w * dx * dx;
            var_y += w * dy * dy;
        }
        st.sigma_x = std::sqrt(std::max(0.0, var_x));
        st.sigma_y = std::sqrt(std::max(0.0, var_y));
    }
    return st;
}

void write_slice_csvs(const UnstructuredMesh2D& mesh,
                      const std::vector<Real>& u,
                      const std::vector<Real>& u_exact,
                      const std::string& prefix) {
    std::vector<std::pair<Real, Index>> stream_pts;
    std::vector<std::pair<Real, Index>> cross_pts;
    const Real band_tol = 0.75 * mesh.h_max_diam();
    for (Index c = 0; c < mesh.num_cells(); ++c) {
        const Vec2& p = mesh.cells()[c].centroid;
        if (std::abs(p.y - 0.5) <= band_tol) stream_pts.emplace_back(p.x, c);
        if (std::abs(p.x - 2.75) <= band_tol) cross_pts.emplace_back(p.y, c);
    }
    std::sort(stream_pts.begin(), stream_pts.end());
    std::sort(cross_pts.begin(), cross_pts.end());

    {
        std::ofstream out(prefix + "_streamwise_y0.5.csv");
        out << std::setprecision(15) << "x,y,u,u_exact\n";
        for (auto [coord, c] : stream_pts) {
            const Vec2& p = mesh.cells()[c].centroid;
            out << p.x << ',' << p.y << ',' << u[c] << ',' << u_exact[c] << '\n';
        }
    }
    {
        std::ofstream out(prefix + "_cross_x2.75.csv");
        out << std::setprecision(15) << "x,y,u,u_exact\n";
        for (auto [coord, c] : cross_pts) {
            const Vec2& p = mesh.cells()[c].centroid;
            out << p.x << ',' << p.y << ',' << u[c] << ',' << u_exact[c] << '\n';
        }
    }
}

void init_patch(const UnstructuredMesh2D& m, Real t_now,
                Real sx0, Real sx1, Real sy0, Real sy1,
                std::vector<Real>& field) {
    field.resize(m.num_cells());
    for (Index c = 0; c < m.num_cells(); ++c) {
        const Real box_area = geom::intersection_area_box(
            m.cell_polygon(c), sx0 + t_now, sx1 + t_now, sy0, sy1);
        field[c] = box_area / m.cells()[c].area;
    }
}

}  // namespace

int main(int argc, char** argv) {
    mpi::MpiGuard mpi_guard(argc, argv);
    const bool is_root = (mpi_guard.rank() == 0);

    const Real lx = 4.0;
    const Real ly = 1.0;
    Real t_end = 2.0;
    Real cfl = 0.45;
    std::string mesh_mode = "nested";  // nested | irregular
    Real tri_jitter = -1.0;
    int tri_smooth = -1;
    std::string scheme_arg = "all";
    std::vector<Index> ref_levels = {0, 1, 2, 3};
    std::string out_dir = "output/problem2";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mesh" && i + 1 < argc) mesh_mode = argv[++i];
        else if (arg == "--tri-jitter" && i + 1 < argc) tri_jitter = std::stod(argv[++i]);
        else if (arg == "--tri-smooth" && i + 1 < argc) tri_smooth = std::stoi(argv[++i]);
        else if (arg == "--scheme" && i + 1 < argc) scheme_arg = argv[++i];
        else if (arg == "--t" && i + 1 < argc) t_end = std::stod(argv[++i]);
        else if (arg == "--cfl" && i + 1 < argc) cfl = std::stod(argv[++i]);
        else if (arg == "--refs" && i + 1 < argc) ref_levels = parse_list(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    }

    if (mesh_mode != "nested" && mesh_mode != "irregular") {
        throw std::invalid_argument("Unknown --mesh mode (expected nested|irregular): " + mesh_mode);
    }

    const auto schemes = solvers::parse_scheme_selection(scheme_arg);
    const Real sx0 = 0.5, sx1 = 1.0, sy0 = 0.25, sy1 = 0.75;
    auto velocity = [](Real, Real) -> Vec2 { return {1.0, 0.0}; };
    auto inflow_bc = [](Real, Real, Real, int) -> Real { return 0.0; };

    const Real eff_jitter = (tri_jitter >= 0.0) ? tri_jitter : (mesh_mode == "irregular" ? 0.45 : 0.35);
    const int eff_smooth = (tri_smooth >= 0) ? tri_smooth : (mesh_mode == "irregular" ? 0 : 5);
    const Index max_ref = *std::max_element(ref_levels.begin(), ref_levels.end());

    std::vector<UnstructuredMesh2D> meshes;
    if (mesh_mode == "nested") {
        meshes.push_back(generate_unstructured_tri_mesh(
            0.0, lx, 0.0, ly, 22, 6, eff_jitter, eff_smooth, 42));
        for (Index r = 0; r < max_ref; ++r) {
            meshes.push_back(meshes.back().refine_uniform());
        }
    } else {
        meshes.reserve(max_ref + 1);
        for (Index r = 0; r <= max_ref; ++r) {
            const Index scale = Index{1} << r;
            meshes.push_back(generate_unstructured_tri_mesh(
                0.0, lx, 0.0, ly, 22 * scale, 6 * scale,
                eff_jitter, eff_smooth, static_cast<unsigned int>(101 + 37 * r)));
        }
    }

    const std::string mesh_suffix = (mesh_mode == "nested") ? "" : ("_" + mesh_mode);
    if (is_root) {
        std::filesystem::create_directories(out_dir);
        std::cout << "Problem 2: Inviscid Tube Transport on Unstructured Triangles (mesh=" << mesh_mode
                  << " [jitter=" << eff_jitter << ", smooth=" << eff_smooth << "], a=(1,0), T=" << t_end
                  << ", CFL=" << cfl << ", ranks=" << mpi_guard.size() << ")\n";
        std::cout << "Exact patch at T=" << t_end << ": S(T)=[" << sx0 + t_end << "," << sx1 + t_end
                  << "] x [" << sy0 << "," << sy1 << "], mass=0.25, sigma_x=sigma_y="
                  << 0.5 / std::sqrt(12.0) << "\n";
    }

    std::vector<ConvergenceRow> rows;
    for (const auto scheme : schemes) {
        std::string label = "1st-Tri";
        std::string tag = "first";
        time::RKMethod rk = time::RKMethod::Euler;
        if (scheme == solvers::ReconstructionScheme::TVD) {
            label = "TVD-Tri";
            tag = "tvd";
            rk = time::RKMethod::SSP_RK2;
        } else if (scheme == solvers::ReconstructionScheme::ENO) {
            label = "ENO3-Tri";
            tag = "eno3";
            rk = time::RKMethod::SSP_RK3;
        } else if (scheme == solvers::ReconstructionScheme::WENO) {
            label = "WENO-Tri";
            tag = "weno";
            rk = time::RKMethod::SSP_RK3;
        }

        if (is_root) {
            std::cout << "\n[" << label << "]\n";
        }

        for (Index ridx = 0; ridx < ref_levels.size(); ++ridx) {
            const Index ref = ref_levels[ridx];
            const auto& mesh = meshes.at(ref);
            if (is_root) {
                mesh.verify_geometry();
            }

            std::vector<Real> u, u_exact;
            init_patch(mesh, 0.0, sx0, sx1, sy0, sy1, u);
            init_patch(mesh, t_end, sx0, sx1, sy0, sy1, u_exact);

            const auto u_decomp = mpi::UnstructuredDecomp::create(mesh);
            solvers::UnstructuredAdvectionSolver solver(
                mesh, velocity, inflow_bc, scheme, 3,
                solvers::UnstructuredLimiter::BarthJespersen, 3,
                solvers::WENOWeightType::WENO_Z, &u_decomp);

            solvers::UnstructuredRunOptions opts;
            opts.scheme = scheme;
            opts.limiter = solvers::UnstructuredLimiter::BarthJespersen;
            opts.eno_order = 3;
            opts.weno_type = solvers::WENOWeightType::WENO_Z;
            opts.rk_method = rk;
            opts.cfl = cfl;
            opts.t_end = t_end;
            opts.output_interval = t_end / 10.0;
            if (ridx + 1 == ref_levels.size()) {
                opts.vtk_dir = out_dir + "/" + tag + mesh_suffix + "_ref" + std::to_string(ref);
            }

            auto exact_diag = [&](Real t_now, const UnstructuredMesh2D& m,
                                  const std::vector<Real>& num,
                                  std::vector<Real>& ex_out, std::vector<Real>& err_out) {
                init_patch(m, t_now, sx0, sx1, sy0, sy1, ex_out);
                err_out.resize(num.size());
                for (Index c = 0; c < num.size(); ++c) {
                    err_out[c] = std::abs(num[c] - ex_out[c]);
                }
            };

            const auto summary = solver.run(u, opts, exact_diag);
            if (is_root) {
                const auto norms = compute_error_norms(u, u_exact, mesh.cell_areas());
                const auto smear = compute_tube_smearing(mesh, u, sx0 + t_end, sx1 + t_end, sy0, sy1);
                const auto smear_ex = compute_tube_smearing(
                    mesh, u_exact, sx0 + t_end, sx1 + t_end, sy0, sy1);
                const Real excess_cross_leak =
                    smear.cross_stream_leakage - smear_ex.cross_stream_leakage;

                if (ridx + 1 == ref_levels.size()) {
                    write_slice_csvs(
                        mesh, u, u_exact,
                        out_dir + "/slice_" + tag + mesh_suffix + "_ref" + std::to_string(ref));
                }

                std::cout << "  [ref=" << ref << "] cells=" << mesh.num_cells()
                          << " | h_diam=" << mesh.h_max_diam()
                          << " | q_min=" << mesh.min_quality()
                          << " | u in [" << summary.final_stats.min_val << ", "
                          << summary.final_stats.max_val << "]"
                          << " | E1=" << norms.l1
                          << " | cross_leak=" << smear.cross_stream_leakage
                          << " (excess=" << excess_cross_leak << ")"
                          << " | stream_spill=" << smear.streamwise_spill
                          << " | sig=(" << smear.sigma_x << "," << smear.sigma_y << ")"
                          << " | outlet_leak=" << summary.outlet_outflow_mass
                          << " | max_RM=" << summary.max_conservation_residual << "\n";

                rows.push_back({
                    label,
                    mesh.num_cells(),
                    mesh.h_max_diam(),
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
        write_convergence_csv(out_dir + "/convergence" + mesh_suffix + ".csv", rows);
        std::cout << "Saved ParaView output and slice profiles to " << out_dir << "/\n";
    }
    return 0;
}
