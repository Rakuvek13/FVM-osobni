#include "fvm/core/convergence.hpp"
#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/io/vtk_writer.hpp"
#include "fvm/linalg/dense.hpp"
#include "fvm/linalg/sparse.hpp"
#include "fvm/mesh/geometry_exact.hpp"
#include "fvm/mesh/mesh_generator.hpp"
#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/mesh/unstructured_mesh.hpp"
#include "fvm/numerics/flux.hpp"
#include "fvm/numerics/quadrature.hpp"
#include "fvm/numerics/time_integrator.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

using namespace fvm;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_quadrature() {
    // 1D Gauss-Legendre up to degree 9 (n=5)
    for (int n = 1; n <= 5; ++n) {
        const auto pts = quadrature::gauss_legendre_01(n);
        const int max_deg = 2 * n - 1;
        for (int p = 0; p <= max_deg; ++p) {
            Real num = 0.0;
            for (const auto& qp : pts) {
                num += qp.weight * std::pow(qp.xi, p);
            }
            const Real exact = 1.0 / static_cast<Real>(p + 1);
            check(std::abs(num - exact) < 1.0e-14, "1D Gauss-Legendre exactness");
        }
    }

    // 2D Triangle quadrature up to degree 5
    const std::vector<int> degrees = {1, 2, 5};
    for (int deg : degrees) {
        const auto pts = quadrature::triangle_rule(deg);
        for (int p = 0; p <= deg; ++p) {
            for (int q = 0; p + q <= deg; ++q) {
                Real num = 0.0;
                for (const auto& qp : pts) {
                    num += qp.weight * std::pow(qp.lambda1, p) * std::pow(qp.lambda2, q);
                }
                // On reference triangle, integral / area = 2! p! q! / (p+q+2)!
                auto fact = [](int k) {
                    Real r = 1.0;
                    for (int i = 2; i <= k; ++i) r *= static_cast<Real>(i);
                    return r;
                };
                const Real exact = 2.0 * fact(p) * fact(q) / fact(p + q + 2);
                check(std::abs(num - exact) < 1.0e-14, "2D triangle quadrature exactness");
            }
        }
    }
    std::cout << "[PASS] quadrature\n";
}

void test_dense_linalg() {
    // LU solve
    linalg::DenseMatrix A(3, 3);
    A(0, 0) = 2.0;  A(0, 1) = 1.0;  A(0, 2) = -1.0;
    A(1, 0) = -3.0; A(1, 1) = -1.0; A(1, 2) = 2.0;
    A(2, 0) = -2.0; A(2, 1) = 1.0;  A(2, 2) = 2.0;
    const std::vector<Real> b = {8.0, -11.0, -3.0};
    const auto x = linalg::solve_lu(A, b);
    check(std::abs(x[0] - 2.0) < 1e-13 &&
          std::abs(x[1] - 3.0) < 1e-13 &&
          std::abs(x[2] + 1.0) < 1e-13, "LU solve");

    // Precomputed QR least squares: recover quadratic polynomial exactly
    linalg::DenseMatrix M(8, 5);
    std::vector<Real> rhs(8);
    const std::vector<Real> true_c = {1.5, -0.7, 0.3, -0.4, 0.9};
    for (Index i = 0; i < 8; ++i) {
        const Real px = 0.2 * static_cast<Real>(i) - 0.5;
        const Real py = std::sin(static_cast<Real>(i) + 0.3);
        M(i, 0) = px;
        M(i, 1) = py;
        M(i, 2) = px * px;
        M(i, 3) = px * py;
        M(i, 4) = py * py;
        rhs[i] = true_c[0] * M(i, 0) + true_c[1] * M(i, 1) +
                 true_c[2] * M(i, 2) + true_c[3] * M(i, 3) + true_c[4] * M(i, 4);
    }
    const auto qr = linalg::QRDecomposition::compute(M);
    const auto c = qr.solve(rhs);
    for (Index j = 0; j < 5; ++j) {
        check(std::abs(c[j] - true_c[j]) < 1e-12, "QR polynomial least squares");
    }
    std::cout << "[PASS] dense_linalg\n";
}

void test_sparse_solvers() {
    // Assemble 2D 5-point Poisson matrix on 16x16 grid (SPD) and solve with CG
    const Index n = 16;
    const Index N = n * n;
    linalg::TripletBuilder tb_spd(N, N, 5 * N);
    std::vector<Real> x_exact(N), b(N, 0.0);
    for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
            const Index id = j * n + i;
            x_exact[id] = std::sin(0.2 * static_cast<Real>(i + 1)) *
                          std::cos(0.15 * static_cast<Real>(j + 1));
            tb_spd.add(id, id, 4.0);
            if (i > 0) tb_spd.add(id, id - 1, -1.0);
            if (i + 1 < n) tb_spd.add(id, id + 1, -1.0);
            if (j > 0) tb_spd.add(id, id - n, -1.0);
            if (j + 1 < n) tb_spd.add(id, id + n, -1.0);
        }
    }
    const auto A_spd = tb_spd.to_csr();
    b = A_spd * x_exact;

    std::vector<Real> x_cg(N, 0.0);
    linalg::LinearSolverConfig cfg;
    cfg.rtol = 1e-12;
    cfg.precond = linalg::Preconditioner::SGS;
    const auto res_cg = linalg::solve_cg(A_spd, b, x_cg, cfg);
    check(res_cg.converged, "CG converged");
    for (Index i = 0; i < N; ++i) {
        check(std::abs(x_cg[i] - x_exact[i]) < 1e-9, "CG solution accuracy");
    }

    // Non-symmetric advection-diffusion matrix solved with BiCGSTAB
    linalg::TripletBuilder tb_ns(N, N, 5 * N);
    for (Index j = 0; j < n; ++j) {
        for (Index i = 0; i < n; ++i) {
            const Index id = j * n + i;
            tb_ns.add(id, id, 5.0);
            if (i > 0) tb_ns.add(id, id - 1, -1.8);
            if (i + 1 < n) tb_ns.add(id, id + 1, -0.4);
            if (j > 0) tb_ns.add(id, id - n, -1.5);
            if (j + 1 < n) tb_ns.add(id, id + n, -0.5);
        }
    }
    const auto A_ns = tb_ns.to_csr();
    const auto b_ns = A_ns * x_exact;
    std::vector<Real> x_bi(N, 0.0);
    const auto res_bi = linalg::solve_bicgstab(A_ns, b_ns, x_bi, cfg);
    check(res_bi.converged, "BiCGSTAB converged");
    for (Index i = 0; i < N; ++i) {
        check(std::abs(x_bi[i] - x_exact[i]) < 1e-9, "BiCGSTAB solution accuracy");
    }
    std::cout << "[PASS] sparse_solvers (CG=" << res_cg.iterations
              << " iters, BiCGSTAB=" << res_bi.iterations << " iters)\n";
}

void test_structured_and_unstructured_meshes() {
    // Structured mesh + ghost cells + cell average initialization
    StructuredMesh2D smesh(0.0, 2.0, 0.0, 1.0, 20, 10, 3);
    auto sf = smesh.create_field();
    auto poly = [](Real x, Real y) { return x * x * y + 3.0 * x - 2.0 * y + 1.0; };
    smesh.initialize_cell_averages(sf, poly, 4);
    smesh.apply_boundary_conditions(sf, StructuredBoundaryConfig::all_periodic(), 0.0);
    check(std::abs(sf(-1, 2) - sf(19, 2)) < 1e-14, "Structured periodic ghost left");
    check(std::abs(sf(20, -2) - sf(0, 8)) < 1e-14, "Structured periodic ghost corner");

    // Total integral of polynomial on [0,2]x[0,1]:
    // int_0^2 int_0^1 (x^2 y + 3x - 2y + 1) dy dx = int_0^2 (0.5 x^2 + 3x) dx = 8/6 + 6 = 22/3
    const auto stats = sf.interior_stats(smesh.cell_area());
    check(std::abs(stats.total_mass - 22.0 / 3.0) < 1e-12, "Structured quadrature mass exactness");

    // Unstructured Delaunay mesh + geometry invariants + stencil collection
    const auto umesh = generate_unstructured_tri_mesh(0.0, 2.0, 0.0, 1.0, 16, 8, 0.35, 5, 123);
    umesh.verify_geometry();
    std::vector<Real> uf;
    umesh.initialize_cell_averages(uf, poly, 5);
    const auto ustats = compute_field_stats(uf, umesh.cell_areas());
    check(std::abs(ustats.total_mass - 22.0 / 3.0) < 1e-12,
          "Unstructured quadrature mass exactness");

    for (Index c = 0; c < umesh.num_cells(); ++c) {
        const auto stencil = umesh.collect_stencil(c, 8, true);
        check(stencil.size() >= 8, "Unstructured stencil min size");
    }

    // Structured tri mesh variant check
    const auto smtri = generate_structured_tri_mesh(0.0, 1.0, 0.0, 1.0, 8, 8,
                                                    StructuredTriPattern::UnionJack);
    smtri.verify_geometry();
    check(smtri.num_cells() == 256, "UnionJack triangle count");

    // Gmsh v2.2 ASCII reader check (including CRLF lines & physical boundary tags)
    std::filesystem::create_directories("build/test_vtk_out");
    const std::string msh_path = "build/test_vtk_out/unit_square.msh";
    {
        std::ofstream msh(msh_path);
        msh << "$MeshFormat\r\n2.2 0 8\r\n$EndMeshFormat\r\n"
            << "$Nodes\r\n4\r\n"
            << "1 0 0 0\r\n2 1 0 0\r\n3 1 1 0\r\n4 0 1 0\r\n$EndNodes\r\n"
            << "$Elements\r\n6\r\n"
            << "1 1 2 10 1 1 2\r\n"
            << "2 1 2 20 2 2 3\r\n"
            << "3 1 2 30 3 3 4\r\n"
            << "4 1 2 40 4 4 1\r\n"
            << "5 2 2 1 1 1 2 3\r\n"
            << "6 2 2 1 1 1 3 4\r\n$EndElements\r\n";
    }
    const auto gmesh = load_gmsh_v2(msh_path);
    gmesh.verify_geometry();
    check(gmesh.num_cells() == 2 && gmesh.num_faces() == 5, "Gmsh v2.2 topology");
    int bnd_tag_sum = 0;
    for (const auto& f : gmesh.faces()) {
        if (f.right_cell < 0) bnd_tag_sum += f.boundary_tag;
    }
    check(bnd_tag_sum == 100, "Gmsh v2.2 boundary physical tags");

    // Uniform 1-to-4 refinement check: h halves and quality is preserved
    const auto refined = umesh.refine_uniform();
    refined.verify_geometry();
    check(refined.num_cells() == 4 * umesh.num_cells(), "Uniform refinement 4x cells");
    check(std::abs(refined.h_max_diam() - 0.5 * umesh.h_max_diam()) < 1e-13,
          "Uniform refinement halves h_max_diam");
    check(std::abs(refined.min_quality() - umesh.min_quality()) < 1e-13,
          "Uniform refinement preserves triangle quality");

    // Exact geometric intersection tests
    // 1. Diagonal band |y - x| <= 0.2 in [0,1]^2 has area 1 - (0.8)^2 = 0.36
    const std::vector<Vec2> unit_sq = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    check(std::abs(geom::intersection_area_diagonal_band(unit_sq, 0.2) - 0.36) < 1e-14,
          "Exact diagonal band area in unit square");
    // Sum over unstructured mesh cells of band intersection equals 0.36
    Real umesh_band_sum = 0.0;
    const auto umesh_unit = generate_unstructured_tri_mesh(0.0, 1.0, 0.0, 1.0, 10, 10, 0.35, 3, 7);
    for (Index c = 0; c < umesh_unit.num_cells(); ++c) {
        umesh_band_sum += geom::intersection_area_diagonal_band(umesh_unit.cell_polygon(c), 0.2);
    }
    check(std::abs(umesh_band_sum - 0.36) < 1e-13, "Band area summed over triangles == 0.36");

    // 2. Disk intersection area summed over grid == pi
    Real disk_sum = 0.0;
    const Index gn = 50;
    const Real gh = 5.0 / static_cast<Real>(gn);
    for (Index j = 0; j < gn; ++j) {
        for (Index i = 0; i < gn; ++i) {
            const Real x0 = -2.5 + static_cast<Real>(i) * gh;
            const Real y0 = -2.5 + static_cast<Real>(j) * gh;
            disk_sum += geom::disk_intersection_area_rect(x0, x0 + gh, y0, y0 + gh, 1.0);
        }
    }
    check(std::abs(disk_sum - PI) < 1e-13, "Exact disk area sum == pi");
    check(std::abs(geom::interval_overlap_fraction(0.1, 0.3, 0.0, 0.2) - 0.5) < 1e-14,
          "Exact 1D overlap fraction");

    std::cout << "[PASS] meshes (unstructured cells=" << umesh.num_cells()
              << ", faces=" << umesh.num_faces() << ")\n";
}

void test_time_integrators_and_flux() {
    // Solve du/dt = -u + cos(t) with u(0) = 1 -> exact u(t) = 0.5*(exp(-t) + sin(t) + cos(t))
    auto exact_u = [](Real t) { return 0.5 * (std::exp(-t) + std::sin(t) + std::cos(t)); };
    auto rhs = [](const std::vector<Real>& u, Real t, std::vector<Real>& dudt) {
        dudt.resize(u.size());
        dudt[0] = -u[0] + std::cos(t);
    };

    auto run_method = [&](time::RKMethod method, Index nsteps) {
        std::vector<Real> u = {1.0};
        time::RKStepper<std::vector<Real>> stepper(method);
        const Real T = 1.0;
        const Real dt = T / static_cast<Real>(nsteps);
        for (Index k = 0; k < nsteps; ++k) {
            stepper.step(u, static_cast<Real>(k) * dt, dt, rhs);
        }
        return std::abs(u[0] - exact_u(T));
    };

    const Real e2_c = run_method(time::RKMethod::SSP_RK2, 20);
    const Real e2_f = run_method(time::RKMethod::SSP_RK2, 40);
    const Real order2 = experimental_order(e2_c, e2_f, 1.0 / 20.0, 1.0 / 40.0);
    check(order2 > 1.95 && order2 < 2.1, "SSP_RK2 2nd order convergence");

    const Real e3_c = run_method(time::RKMethod::SSP_RK3, 20);
    const Real e3_f = run_method(time::RKMethod::SSP_RK3, 40);
    const Real order3 = experimental_order(e3_c, e3_f, 1.0 / 20.0, 1.0 / 40.0);
    check(order3 > 2.95 && order3 < 3.1, "SSP_RK3 3rd order convergence");

    const Real e4_c = run_method(time::RKMethod::RK4, 10);
    const Real e4_f = run_method(time::RKMethod::RK4, 20);
    const Real order4 = experimental_order(e4_c, e4_f, 1.0 / 10.0, 1.0 / 20.0);
    check(order4 > 3.95 && order4 < 4.1, "RK4 4th order convergence");

    // Nonconvex wave speed test (KPP: f(u) = sin(u), f'(u) = cos(u))
    // Between uL = pi/2 and uR = 3*pi/2, endpoints have cos = 0, but midpoint pi has |cos| = 1!
    const Real alpha = flux::interval_max_wave_speed(
        PI * 0.5, PI * 1.5, [](Real u) { return std::cos(u); }, 16);
    check(alpha > 0.99, "Nonconvex wave speed detects interior extremum");
    std::cout << "[PASS] time_integrators (RK2=" << order2
              << ", RK3=" << order3 << ", RK4=" << order4 << ") & flux\n";
}

void test_vtk_output() {
    const std::filesystem::path tmp_dir = "build/test_vtk_out";
    std::filesystem::remove_all(tmp_dir);

    StructuredMesh2D smesh(0.0, 1.0, 0.0, 1.0, 8, 8, 1);
    auto sf = smesh.create_field();
    smesh.initialize_cell_averages(sf, [](Real x, Real y) { return std::sin(PI * x) * std::cos(PI * y); });
    io::write_vti((tmp_dir / "struct_0000.vti").string(), smesh, {{"u", &sf}});

    const auto umesh = generate_unstructured_tri_mesh(0.0, 2.0, 0.0, 1.0, 8, 4);
    std::vector<Real> uf;
    umesh.initialize_cell_averages(uf, [](Real x, Real y) { return x + y; });
    std::vector<Vec2> vel(umesh.num_cells(), Vec2{1.0, 0.0});
    io::write_vtu((tmp_dir / "unstruct_0000.vtu").string(), umesh, {{"u", &uf}}, {{"velocity", &vel}});

    io::PvdWriter pvd((tmp_dir / "series.pvd").string());
    pvd.add_step(0.0, "unstruct_0000.vtu");
    pvd.save();

    check(std::filesystem::exists(tmp_dir / "struct_0000.vti"), "VTI file created");
    check(std::filesystem::exists(tmp_dir / "unstruct_0000.vtu"), "VTU file created");
    check(std::filesystem::exists(tmp_dir / "series.pvd"), "PVD file created");

    // Smoke test convergence table formatting
    std::vector<ConvergenceRow> rows = {
        {"1st-Order", 100, 0.1, {1.0e-1, 1.2e-1, 2.0e-1}, 0, 0, 0, 1e-15, 0.01},
        {"1st-Order", 400, 0.05, {5.0e-2, 6.0e-2, 1.0e-1}, 0, 0, 0, 1e-15, 0.04}
    };
    populate_eoc(rows);
    check(std::abs(rows[1].eoc_l1 - 1.0) < 1e-12, "EOC calculation");
    std::ostringstream oss;
    print_convergence_table(oss, rows);
    check(!oss.str().empty(), "Convergence table formatting");

    std::cout << "[PASS] vtk_output & convergence\n";
}

}  // namespace

int main(int argc, char** argv) {
    mpi::MpiGuard mpi_guard(argc, argv);
    if (mpi_guard.rank() == 0) {
        test_quadrature();
        test_dense_linalg();
        test_sparse_solvers();
        test_structured_and_unstructured_meshes();
        test_time_integrators_and_flux();
        test_vtk_output();
        std::cout << "All core infrastructure tests passed.\n";
    }
    return 0;
}
