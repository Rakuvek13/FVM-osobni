#include "fvm/core/convergence.hpp"
#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/io/vtk_writer.hpp"
#include "fvm/mesh/geometry_exact.hpp"
#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/numerics/flux.hpp"
#include "fvm/numerics/quadrature.hpp"
#include "fvm/numerics/time_integrator.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
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
    std::cout << "[PASS] quadrature\n";
}

void test_structured_meshes_and_geometry() {
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

    // Exact geometric intersection tests
    // 1. Diagonal band |y - x| <= 0.2 in [0,1]^2 has area 1 - (0.8)^2 = 0.36
    const std::vector<Vec2> unit_sq = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    check(std::abs(geom::intersection_area_diagonal_band(unit_sq, 0.2) - 0.36) < 1e-14,
          "Exact diagonal band area in unit square");

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

    std::cout << "[PASS] structured_meshes_and_geometry\n";
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

    io::PvdWriter pvd((tmp_dir / "series.pvd").string());
    pvd.add_step(0.0, "struct_0000.vti");
    pvd.save();

    check(std::filesystem::exists(tmp_dir / "struct_0000.vti"), "VTI file created");
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
        test_structured_meshes_and_geometry();
        test_time_integrators_and_flux();
        test_vtk_output();
        std::cout << "All core infrastructure tests passed.\n";
    }
    return 0;
}
