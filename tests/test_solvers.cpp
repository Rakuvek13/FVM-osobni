#include "fvm/core/convergence.hpp"
#include "fvm/core/field.hpp"
#include "fvm/core/mpi_context.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/geometry_exact.hpp"
#include "fvm/mesh/mesh_generator.hpp"
#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/mesh/unstructured_mesh.hpp"
#include "fvm/solvers/structured_solver.hpp"
#include "fvm/solvers/unstructured_solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using namespace fvm;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_diagonal_band_and_aligned_control() {
    // Aligned control: horizontal band |y - 0.5| < 0.2 with a = (1, 0) must stay exact (E1 ~ 0)
    // when w aligns with faces (N = 40 -> dy = 0.025, 0.3 and 0.7 are exact multiples!).
    const Real w = 0.2;
    const Index N = 40;
    StructuredMesh2D mesh(0.0, 1.0, 0.0, 1.0, N, N, 1);
    auto u = mesh.create_field();
    for (Index j = 0; j < N; ++j) {
        const Real y0 = static_cast<Real>(j) * mesh.dy();
        const Real y1 = y0 + mesh.dy();
        const Real frac = geom::interval_overlap_fraction(y0, y1, 0.5 - w, 0.5 + w);
        for (Index i = 0; i < N; ++i) {
            u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = frac;
        }
    }
    const auto u_init = u;

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletFaceAvg;
    bc.left.face_avg = [w](Real y0, Real y1, Real) {
        return geom::interval_overlap_fraction(y0, y1, 0.5 - w, 0.5 + w);
    };
    bc.right.type = StructuredBCType::ZeroGradient;
    bc.bottom.type = StructuredBCType::ZeroGradient;
    bc.top.type = StructuredBCType::ZeroGradient;

    solvers::StructuredFVSolver solver(
        mesh, bc, solvers::make_linear_advection_law(1.0, 0.0),
        solvers::ReconstructionScheme::FirstOrder);
    solvers::SolverRunOptions opts;
    opts.rk_method = time::RKMethod::Euler;
    opts.cfl = 0.45;
    opts.t_end = 1.0;
    const auto sum = solver.run(u, opts);
    const auto norms = compute_error_norms(u, u_init, mesh.cell_area());
    check(norms.l1 < 1.0e-14, "Aligned grid control has zero crosswind diffusion");
    check(sum.max_conservation_residual < 1.0e-13, "Problem 1 aligned conservation residual");

    // Diagonal smooth stationary band EOC test
    auto smooth_band = [w](Real x, Real y) {
        const Real d = (y - x) / w;
        return std::exp(-d * d);
    };
    auto run_diag = [&](Index n) {
        StructuredMesh2D m(0.0, 1.0, 0.0, 1.0, n, n, 1);
        auto f = m.create_field();
        auto f_ex = m.create_field();
        m.initialize_cell_averages(f, smooth_band, 4);
        m.initialize_cell_averages(f_ex, smooth_band, 4);
        StructuredBoundaryConfig dbc;
        dbc.left.type = StructuredBCType::DirichletExact;
        dbc.left.value = [&](Real x, Real y, Real) { return smooth_band(x, y); };
        dbc.bottom.type = StructuredBCType::DirichletExact;
        dbc.bottom.value = [&](Real x, Real y, Real) { return smooth_band(x, y); };
        dbc.right.type = StructuredBCType::ZeroGradient;
        dbc.top.type = StructuredBCType::ZeroGradient;

        solvers::StructuredFVSolver s(
            m, dbc, solvers::make_linear_advection_law(1.0, 1.0),
            solvers::ReconstructionScheme::FirstOrder);
        solvers::SolverRunOptions o;
        o.cfl = 0.45;
        o.t_end = 1.0;
        const auto res = s.run(f, o);
        check(res.max_conservation_residual < 1.0e-13, "Diagonal smooth conservation residual");
        return compute_error_norms(f, f_ex, m.cell_area());
    };
    const auto e1 = run_diag(40);
    const auto e2 = run_diag(80);
    const Real eoc = experimental_order(e1.l1, e2.l1, 1.0 / 40.0, 1.0 / 80.0);
    check(eoc > 0.55 && eoc < 1.1, "Diagonal stationary smooth band 1st-order EOC");
    std::cout << "[PASS] diagonal_band_and_aligned_control (smooth EOC_L1=" << eoc << ")\n";
}

void test_unstructured_tube_conservation_and_bounds() {
    const auto base = generate_unstructured_tri_mesh(0.0, 4.0, 0.0, 1.0, 22, 6, 0.35, 5, 42);
    const auto mesh = base.refine_uniform();
    mesh.verify_geometry();

    std::vector<Real> u(mesh.num_cells(), 0.0);
    for (Index c = 0; c < mesh.num_cells(); ++c) {
        u[c] = geom::intersection_area_box(mesh.cell_polygon(c), 0.5, 1.0, 0.25, 0.75) /
               mesh.cells()[c].area;
    }

    solvers::UnstructuredAdvectionSolver solver(
        mesh, [](Real, Real) { return Vec2{1.0, 0.0}; },
        [](Real, Real, Real, int) { return 0.0; },
        solvers::ReconstructionScheme::FirstOrder, 3);
    check(solver.max_discrete_divergence() < 1.0e-13, "Tube velocity discrete div == 0");

    solvers::UnstructuredRunOptions opts;
    opts.rk_method = time::RKMethod::Euler;
    opts.cfl = 0.45;
    opts.t_end = 2.0;
    const auto sum = solver.run(u, opts);

    check(sum.initial_stats.total_mass > 0.25 - 1e-13 && sum.initial_stats.total_mass < 0.25 + 1e-13,
          "Exact initial patch mass == 0.25");
    check(sum.final_stats.min_val >= -1.0e-14 && sum.final_stats.max_val <= 1.0 + 1.0e-14,
          "Tube maximum principle [0, 1]");
    check(sum.max_conservation_residual < 1.0e-13, "Tube per-step conservation residual R_M^n ~ 0");
    check(sum.cumulative_conservation_residual < 1.0e-13, "Tube cumulative conservation balance");
    std::cout << "[PASS] unstructured_tube_conservation_and_bounds (max_RM="
              << sum.max_conservation_residual << ")\n";
}

void test_kpp_pdf4_bounds_and_conservation() {
    StructuredMesh2D mesh(-2.5, 2.5, -2.5, 2.5, 100, 100, 1);
    auto u = mesh.create_field();
    const Real u_in = 3.5 * PI;
    const Real u_out = 0.25 * PI;

    for (Index j = 0; j < mesh.ny(); ++j) {
        const Real y0 = mesh.ymin() + static_cast<Real>(j) * mesh.dy();
        for (Index i = 0; i < mesh.nx(); ++i) {
            const Real x0 = mesh.xmin() + static_cast<Real>(i) * mesh.dx();
            const Real frac = geom::disk_intersection_area_rect(
                                  x0, x0 + mesh.dx(), y0, y0 + mesh.dy(), 1.0) /
                              mesh.cell_area();
            u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) =
                u_out + (u_in - u_out) * frac;
        }
    }

    solvers::StructuredConservationLaw law;
    law.fx = [](Real, Real, Real v) { return std::sin(v); };
    law.fy = [](Real, Real, Real v) { return std::cos(v); };
    law.alpha_x = [](Real, Real, Real, Real) { return 1.0; };
    law.alpha_y = [](Real, Real, Real, Real) { return 1.0; };
    law.max_speeds = [](const StructuredField2D&, Real& ax, Real& ay) { ax = 1.0; ay = 1.0; };

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletExact;
    bc.left.value = [u_out](Real, Real, Real) { return u_out; };
    bc.right = bc.left;
    bc.bottom = bc.left;
    bc.top = bc.left;

    solvers::StructuredFVSolver solver(mesh, bc, law, solvers::ReconstructionScheme::FirstOrder);
    solvers::SolverRunOptions opts;
    opts.cfl = 0.45;
    opts.t_end = 1.0;
    const auto sum = solver.run(u, opts);

    check(sum.final_stats.min_val >= u_out - 1.0e-12, "KPP min >= pi/4");
    check(sum.final_stats.max_val <= u_in + 1.0e-12, "KPP max <= 7pi/2");
    check(sum.max_conservation_residual < 1.0e-12, "KPP conservation residual R_M^n ~ 0");
    std::cout << "[PASS] kpp_pdf4_bounds_and_conservation (max_RM="
              << sum.max_conservation_residual << ")\n";
}

void test_structured_tvd_and_eoc() {
    // 1. Aligned stationary band stays machine-exact under TVD + SSP_RK2
    {
        StructuredMesh2D mesh(0.0, 1.0, 0.0, 1.0, 40, 40, 2);
        auto u = mesh.create_field();
        auto u_exact = mesh.create_field();
        for (Index j = 0; j < mesh.ny(); ++j) {
            const Real y0 = static_cast<Real>(j) * mesh.dy();
            const Real y1 = y0 + mesh.dy();
            const Real frac = geom::interval_overlap_fraction(y0, y1, 0.3, 0.7);
            for (Index i = 0; i < mesh.nx(); ++i) {
                u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = frac;
                u_exact(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = frac;
            }
        }
        StructuredBoundaryConfig bc;
        bc.left.type = StructuredBCType::DirichletFaceAvg;
        bc.left.face_avg = [](Real y0, Real y1, Real) {
            return geom::interval_overlap_fraction(y0, y1, 0.3, 0.7);
        };
        bc.right.type = StructuredBCType::ZeroGradient;
        bc.bottom.type = StructuredBCType::ZeroGradient;
        bc.top.type = StructuredBCType::ZeroGradient;

        solvers::StructuredFVSolver solver(
            mesh, bc, solvers::make_linear_advection_law(1.0, 0.0),
            solvers::ReconstructionScheme::TVD, solvers::TVDSlopeLimiter::MC);
        solvers::SolverRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::TVD;
        opts.rk_method = time::RKMethod::SSP_RK2;
        opts.cfl = 0.45;
        opts.t_end = 1.0;
        const auto sum = solver.run(u, opts);
        const auto err = compute_error_norms(u, u_exact, mesh.cell_area());
        check(err.l1 < 1.0e-13, "TVD aligned control exact E1 < 1e-13");
        check(sum.max_conservation_residual < 1.0e-13, "TVD aligned conservation R_M^n ~ 0");
    }

    // 2. Smooth periodic diagonal sine wave achieves ~2nd-order EOC and TVD bounds
    auto smooth_periodic_fn = [](Real x, Real y) -> Real {
        return 0.5 + 0.5 * std::sin(2.0 * PI * (y - x));
    };
    const auto bc_per = StructuredBoundaryConfig::all_periodic();
    const auto law_diag = solvers::make_linear_advection_law(1.0, 1.0);

    auto eval_periodic = [&](Index n, solvers::ReconstructionScheme scheme) -> Real {
        StructuredMesh2D mesh(0.0, 1.0, 0.0, 1.0, n, n, 2);
        auto u = mesh.create_field();
        auto u_exact = mesh.create_field();
        mesh.initialize_cell_averages(u, smooth_periodic_fn, 4);
        u_exact = u;
        solvers::StructuredFVSolver solver(mesh, bc_per, law_diag, scheme, solvers::TVDSlopeLimiter::MC);
        solvers::SolverRunOptions opts;
        opts.scheme = scheme;
        opts.rk_method = (scheme == solvers::ReconstructionScheme::TVD)
            ? time::RKMethod::SSP_RK2 : time::RKMethod::Euler;
        opts.cfl = 0.45;
        opts.t_end = 1.0;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-12, "Structured TVD conservation residual ~ 0");
        check(sum.final_stats.min_val >= -1.0e-12, "Structured TVD min >= 0");
        check(sum.final_stats.max_val <= 1.0 + 1.0e-12, "Structured TVD max <= 1");
        return compute_error_norms(u, u_exact, mesh.cell_area()).l1;
    };

    const Real e1_first_40 = eval_periodic(40, solvers::ReconstructionScheme::FirstOrder);
    const Real e1_tvd_40 = eval_periodic(40, solvers::ReconstructionScheme::TVD);
    const Real e1_tvd_80 = eval_periodic(80, solvers::ReconstructionScheme::TVD);
    const Real eoc_tvd = experimental_order(e1_tvd_40, e1_tvd_80, 1.0 / 40.0, 1.0 / 80.0);
    check(e1_tvd_40 < 0.25 * e1_first_40, "Structured TVD beats 1st-order by >4x at N=40");
    check(eoc_tvd > 1.75, "Structured TVD smooth periodic EOC_L1 > 1.75");

    std::cout << "[PASS] structured_tvd_and_eoc (E1_40: 1st=" << e1_first_40
              << ", TVD=" << e1_tvd_40 << ", EOC_L1=" << eoc_tvd << ")\n";
}

void test_unstructured_tvd() {
    const UnstructuredMesh2D base = generate_unstructured_tri_mesh(
        0.0, 4.0, 0.0, 1.0, 16, 4, 0.35, 5, 42);
    const UnstructuredMesh2D fine = base.refine_uniform();
    auto velocity = [](Real, Real) -> Vec2 { return {1.0, 0.0}; };

    // 1. Free-stream preservation u == 1
    {
        std::vector<Real> u(base.num_cells(), 1.0);
        solvers::UnstructuredAdvectionSolver solver(
            base, velocity, [](Real, Real, Real, int) { return 1.0; },
            solvers::ReconstructionScheme::TVD, 2, solvers::UnstructuredLimiter::BarthJespersen);
        solvers::UnstructuredRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::TVD;
        opts.rk_method = time::RKMethod::SSP_RK2;
        opts.cfl = 0.45;
        opts.t_end = 0.5;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-13, "Unstructured TVD freestream conservation");
        for (Real val : u) {
            check(std::abs(val - 1.0) < 1.0e-12, "Unstructured TVD freestream u == 1");
        }
    }

    // 2. Patch boundedness [0, 1] with Barth-Jespersen and improved L1 accuracy over 1st order
    {
        auto init_patch = [](const UnstructuredMesh2D& m, Real t_now) {
            std::vector<Real> f(m.num_cells());
            for (Index c = 0; c < m.num_cells(); ++c) {
                f[c] = geom::intersection_area_box(
                           m.cell_polygon(c), 0.5 + t_now, 1.0 + t_now, 0.25, 0.75) /
                       m.cells()[c].area;
            }
            return f;
        };
        const Real T = 0.5;
        auto u1 = init_patch(base, 0.0);
        auto u2 = u1;
        const auto u_exact = init_patch(base, T);

        solvers::UnstructuredAdvectionSolver s1(
            base, velocity, [](Real, Real, Real, int) { return 0.0; },
            solvers::ReconstructionScheme::FirstOrder, 3);
        solvers::UnstructuredRunOptions o1;
        o1.rk_method = time::RKMethod::Euler;
        o1.cfl = 0.45;
        o1.t_end = T;
        s1.run(u1, o1);

        solvers::UnstructuredAdvectionSolver s2(
            base, velocity, [](Real, Real, Real, int) { return 0.0; },
            solvers::ReconstructionScheme::TVD, 2, solvers::UnstructuredLimiter::BarthJespersen);
        solvers::UnstructuredRunOptions o2;
        o2.scheme = solvers::ReconstructionScheme::TVD;
        o2.rk_method = time::RKMethod::SSP_RK2;
        o2.cfl = 0.45;
        o2.t_end = T;
        const auto sum2 = s2.run(u2, o2);

        const Real e1_first = compute_error_norms(u1, u_exact, base.cell_areas()).l1;
        const Real e1_tvd = compute_error_norms(u2, u_exact, base.cell_areas()).l1;
        check(sum2.final_stats.min_val >= -1.0e-12, "Unstructured TVD BJ min >= 0");
        check(sum2.final_stats.max_val <= 1.0 + 1.0e-12, "Unstructured TVD BJ max <= 1");
        check(sum2.max_conservation_residual < 1.0e-13, "Unstructured TVD BJ conservation R_M^n ~ 0");
        check(e1_tvd < e1_first, "Unstructured TVD BJ reduces E1 vs 1st order");
    }

    // 3. Smooth bump EOC on unstructured triangles with Venkatakrishnan TVD + SSP_RK2
    auto smooth_ic = [](Real x, Real y) -> Real {
        const Real dx = x - 0.75;
        const Real dy = y - 0.5;
        const Real r = std::hypot(dx, dy);
        const Real R = 0.42;
        if (r >= R) return 0.0;
        const Real c = std::cos(0.5 * PI * r / R);
        const Real c2 = c * c;
        return c2 * c2;
    };
    auto eval_smooth_tri = [&](const UnstructuredMesh2D& m) -> Real {
        const Real T = 0.5;
        std::vector<Real> u, u_ex;
        m.initialize_cell_averages(u, [&](Real x, Real y) { return smooth_ic(x, y); }, 5);
        m.initialize_cell_averages(u_ex, [&](Real x, Real y) { return smooth_ic(x - T, y); }, 5);
        solvers::UnstructuredAdvectionSolver solver(
            m, velocity, [&](Real x, Real y, Real t, int) { return smooth_ic(x - t, y); },
            solvers::ReconstructionScheme::TVD, 3, solvers::UnstructuredLimiter::Venkatakrishnan);
        solvers::UnstructuredRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::TVD;
        opts.rk_method = time::RKMethod::SSP_RK2;
        opts.cfl = 0.45;
        opts.t_end = T;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-13, "Unstructured TVD Venkat conservation");
        return compute_error_norms(u, u_ex, m.cell_areas()).l1;
    };

    const Real e_fine = eval_smooth_tri(fine);
    const UnstructuredMesh2D fine2 = fine.refine_uniform();
    const Real e_fine2 = eval_smooth_tri(fine2);
    const Real eoc_12 = experimental_order(
        e_fine, e_fine2, fine.h_max_diam(), fine2.h_max_diam());
    check(eoc_12 > 1.75, "Unstructured TVD smooth EOC_L1 > 1.75");

    std::cout << "[PASS] unstructured_tvd (E1_fine=" << e_fine
              << ", E1_fine2=" << e_fine2 << ", smooth EOC_L1=" << eoc_12 << ")\n";
}

void test_kpp_tvd() {
    const Real u_in = 3.5 * PI;
    const Real u_out = 0.25 * PI;
    StructuredMesh2D mesh(-2.5, 2.5, -2.5, 2.5, 100, 100, 2);
    auto u = mesh.create_field();
    const Real dx = mesh.dx(), dy = mesh.dy(), area = mesh.cell_area();
    for (Index j = 0; j < mesh.ny(); ++j) {
        const Real y0 = mesh.ymin() + static_cast<Real>(j) * dy;
        for (Index i = 0; i < mesh.nx(); ++i) {
            const Real x0 = mesh.xmin() + static_cast<Real>(i) * dx;
            const Real frac = geom::disk_intersection_area_rect(x0, x0 + dx, y0, y0 + dy, 1.0) / area;
            u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = u_out + (u_in - u_out) * frac;
        }
    }

    solvers::StructuredConservationLaw law;
    law.fx = [](Real, Real, Real v) { return std::sin(v); };
    law.fy = [](Real, Real, Real v) { return std::cos(v); };
    law.alpha_x = [](Real, Real, Real, Real) { return 1.0; };
    law.alpha_y = [](Real, Real, Real, Real) { return 1.0; };
    law.max_speeds = [](const StructuredField2D&, Real& ax, Real& ay) { ax = 1.0; ay = 1.0; };

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletExact;
    bc.left.value = [u_out](Real, Real, Real) { return u_out; };
    bc.right = bc.left;
    bc.bottom = bc.left;
    bc.top = bc.left;

    solvers::StructuredFVSolver solver(
        mesh, bc, law, solvers::ReconstructionScheme::TVD, solvers::TVDSlopeLimiter::Minmod);
    solvers::SolverRunOptions opts;
    opts.scheme = solvers::ReconstructionScheme::TVD;
    opts.rk_method = time::RKMethod::SSP_RK2;
    opts.cfl = 0.45;
    opts.t_end = 1.0;
    const auto sum = solver.run(u, opts);
    check(sum.final_stats.min_val >= u_out - 1.0e-12, "KPP TVD min >= pi/4");
    check(sum.final_stats.max_val <= u_in + 1.0e-12, "KPP TVD max <= 7pi/2");
    check(sum.max_conservation_residual < 1.0e-12, "KPP TVD conservation residual R_M^n ~ 0");

    std::cout << "[PASS] kpp_tvd (max_RM=" << sum.max_conservation_residual << ")\n";
}

void test_structured_eno_and_eoc() {
    // 1. Aligned stationary band stays exact under ENO3 + SSP_RK3
    {
        StructuredMesh2D mesh(0.0, 1.0, 0.0, 1.0, 40, 40, 3);
        auto u = mesh.create_field();
        auto u_exact = mesh.create_field();
        for (Index j = 0; j < mesh.ny(); ++j) {
            const Real y0 = static_cast<Real>(j) * mesh.dy();
            const Real y1 = y0 + mesh.dy();
            const Real frac = geom::interval_overlap_fraction(y0, y1, 0.3, 0.7);
            for (Index i = 0; i < mesh.nx(); ++i) {
                u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = frac;
                u_exact(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = frac;
            }
        }
        StructuredBoundaryConfig bc;
        bc.left.type = StructuredBCType::DirichletFaceAvg;
        bc.left.face_avg = [](Real y0, Real y1, Real) {
            return geom::interval_overlap_fraction(y0, y1, 0.3, 0.7);
        };
        bc.right.type = StructuredBCType::ZeroGradient;
        bc.bottom.type = StructuredBCType::ZeroGradient;
        bc.top.type = StructuredBCType::ZeroGradient;

        solvers::StructuredFVSolver solver(
            mesh, bc, solvers::make_linear_advection_law(1.0, 0.0),
            solvers::ReconstructionScheme::ENO, solvers::TVDSlopeLimiter::MC, 3);
        solvers::SolverRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::ENO;
        opts.eno_order = 3;
        opts.rk_method = time::RKMethod::SSP_RK3;
        opts.cfl = 0.45;
        opts.t_end = 1.0;
        const auto sum = solver.run(u, opts);
        const auto err = compute_error_norms(u, u_exact, mesh.cell_area());
        check(err.l1 < 1.0e-13, "ENO3 aligned control exact E1 < 1e-13");
        check(sum.max_conservation_residual < 1.0e-13, "ENO3 aligned conservation R_M^n ~ 0");
    }

    // 2. Smooth periodic diagonal wave: verify 2nd-order ENO2 and 3rd-order ENO3 in L1, L2, Linf
    auto smooth_periodic_fn = [](Real x, Real y) -> Real {
        return 0.5 + 0.5 * std::sin(2.0 * PI * (y - x));
    };
    const auto bc_per = StructuredBoundaryConfig::all_periodic();
    const auto law_diag = solvers::make_linear_advection_law(1.0, 1.0);

    auto eval_periodic = [&](Index n, int order) -> ErrorNorms {
        StructuredMesh2D mesh(0.0, 1.0, 0.0, 1.0, n, n, 3);
        auto u = mesh.create_field();
        auto u_exact = mesh.create_field();
        mesh.initialize_cell_averages(u, smooth_periodic_fn, 4);
        u_exact = u;
        solvers::StructuredFVSolver solver(
            mesh, bc_per, law_diag, solvers::ReconstructionScheme::ENO,
            solvers::TVDSlopeLimiter::MC, order);
        solvers::SolverRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::ENO;
        opts.eno_order = order;
        opts.rk_method = (order == 2) ? time::RKMethod::SSP_RK2 : time::RKMethod::SSP_RK3;
        opts.cfl = 0.45;
        opts.t_end = 1.0;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-12, "Structured ENO conservation residual ~ 0");
        return compute_error_norms(u, u_exact, mesh.cell_area());
    };

    const auto err2_40 = eval_periodic(40, 2);
    const auto err2_80 = eval_periodic(80, 2);
    const Real eoc2_l1 = experimental_order(err2_40.l1, err2_80.l1, 1.0 / 40.0, 1.0 / 80.0);
    const Real eoc2_linf = experimental_order(err2_40.linf, err2_80.linf, 1.0 / 40.0, 1.0 / 80.0);

    const auto err3_40 = eval_periodic(40, 3);
    const auto err3_80 = eval_periodic(80, 3);
    const Real eoc3_l1 = experimental_order(err3_40.l1, err3_80.l1, 1.0 / 40.0, 1.0 / 80.0);
    const Real eoc3_linf = experimental_order(err3_40.linf, err3_80.linf, 1.0 / 40.0, 1.0 / 80.0);
    check(eoc2_l1 > 1.25, "ENO2 smooth periodic EOC_L1 > 1.25");
    check(eoc2_linf > 1.15, "ENO2 smooth periodic EOC_Linf > 1.15");
    check(err3_80.l1 < err2_80.l1, "ENO3 improves over ENO2 on smooth periodic wave");
    check(eoc3_l1 > 2.75, "ENO3 smooth periodic EOC_L1 > 2.75");
    check(eoc3_linf > 2.5, "ENO3 smooth periodic EOC_Linf > 2.5");

    // 3. Nonlinear 2D KPP smooth short-time test (verifies 2D face Gauss quadrature retains O(h^3))
    {
        auto smooth_nl_ic = [](Real x, Real y) -> Real {
            return 1.0 + 0.25 * std::sin(2.0 * PI * x) * std::sin(2.0 * PI * y);
        };
        solvers::StructuredConservationLaw nl_law;
        nl_law.fx = [](Real, Real, Real v) { return std::sin(v); };
        nl_law.fy = [](Real, Real, Real v) { return std::cos(v); };
        nl_law.alpha_x = [](Real, Real, Real, Real) { return 1.0; };
        nl_law.alpha_y = [](Real, Real, Real, Real) { return 1.0; };
        nl_law.max_speeds = [](const StructuredField2D&, Real& ax, Real& ay) { ax = 1.0; ay = 1.0; };

        auto run_nl = [&](Index n) -> StructuredField2D {
            StructuredMesh2D m(0.0, 1.0, 0.0, 1.0, n, n, 3);
            auto f = m.create_field();
            m.initialize_cell_averages(f, smooth_nl_ic, 5);
            solvers::StructuredFVSolver s(
                m, bc_per, nl_law, solvers::ReconstructionScheme::ENO,
                solvers::TVDSlopeLimiter::MC, 3);
            solvers::SolverRunOptions o;
            o.scheme = solvers::ReconstructionScheme::ENO;
            o.eno_order = 3;
            o.rk_method = time::RKMethod::SSP_RK3;
            o.cfl = 0.4;
            o.t_end = 0.04;
            s.run(f, o);
            return f;
        };

        auto avg_down = [](const StructuredField2D& fine, Index nc) {
            const Index rx = fine.nx() / nc;
            StructuredField2D c(nc, nc, 0, 0.0);
            const Real inv = 1.0 / static_cast<Real>(rx * rx);
            for (Index j = 0; j < nc; ++j) {
                for (Index i = 0; i < nc; ++i) {
                    Real s = 0.0;
                    for (Index sj = 0; sj < rx; ++sj) {
                        for (Index si = 0; si < rx; ++si) {
                            s += fine(static_cast<SignedIndex>(i * rx + si),
                                      static_cast<SignedIndex>(j * rx + sj));
                        }
                    }
                    c(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = s * inv;
                }
            }
            return c;
        };

        const auto ref = run_nl(160);
        const auto f20 = run_nl(20);
        const auto f40 = run_nl(40);
        const Real nl_e20 = compute_error_norms(f20, avg_down(ref, 20), 1.0 / 400.0).l1;
        const Real nl_e40 = compute_error_norms(f40, avg_down(ref, 40), 1.0 / 1600.0).l1;
        const Real nl_eoc = experimental_order(nl_e20, nl_e40, 1.0 / 20.0, 1.0 / 40.0);
        check(nl_eoc > 2.6, "2D nonlinear ENO3 with 2-pt face Gauss exceeds O(h^2) barrier (EOC > 2.6)");
        std::cout << "[PASS] structured_eno_and_eoc (ENO2 EOC_1=" << eoc2_l1
                  << ", ENO3 EOC_1=" << eoc3_l1 << ", Linf_EOC=" << eoc3_linf
                  << ", nonlinear_2D_EOC=" << nl_eoc << ")\n";
    }
}

void test_unstructured_eno() {
    const UnstructuredMesh2D base = generate_unstructured_tri_mesh(
        0.0, 4.0, 0.0, 1.0, 16, 4, 0.35, 5, 42);
    const UnstructuredMesh2D fine = base.refine_uniform();
    const UnstructuredMesh2D fine2 = fine.refine_uniform();
    auto velocity = [](Real, Real) -> Vec2 { return {1.0, 0.0}; };

    // 1. Exact quadratic reproduction P^2(x - t, y) on unstructured triangles with ENO3 + SSP_RK3
    {
        auto p2_fn = [](Real x, Real y) -> Real {
            return 1.0 + 0.2 * x - 0.3 * y + 0.1 * x * x - 0.15 * x * y + 0.08 * y * y;
        };
        const Real T = 0.4;
        std::vector<Real> u, u_ex;
        base.initialize_cell_averages(u, [&](Real x, Real y) { return p2_fn(x, y); }, 5);
        base.initialize_cell_averages(u_ex, [&](Real x, Real y) { return p2_fn(x - T, y); }, 5);
        solvers::UnstructuredAdvectionSolver solver(
            base, velocity, [&](Real x, Real y, Real t, int) { return p2_fn(x - t, y); },
            solvers::ReconstructionScheme::ENO, 3, solvers::UnstructuredLimiter::BarthJespersen, 3);
        // Check exact spatial RHS at t = 0 (zero RK stage mismatch)
        std::vector<Real> dudt, dudt_ex;
        solver.compute_rhs(u, 0.0, dudt);
        base.initialize_cell_averages(dudt_ex, [](Real x, Real y) { return -(0.2 + 0.2 * x - 0.15 * y); }, 5);
        Real max_rhs_err = 0.0;
        for (Index c = 0; c < base.num_cells(); ++c) {
            max_rhs_err = std::max(max_rhs_err, std::abs(dudt[c] - dudt_ex[c]));
        }
        solvers::UnstructuredRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::ENO;
        opts.eno_order = 3;
        opts.rk_method = time::RKMethod::SSP_RK3;
        opts.cfl = 0.45;
        opts.t_end = T;
        const auto sum = solver.run(u, opts);
        const auto err = compute_error_norms(u, u_ex, base.cell_areas());
        check(max_rhs_err < 1.0e-11, "Unstructured ENO3 exact spatial RHS for P2");
        check(err.linf < 1.0e-3, "Unstructured ENO3 + SSP_RK3 low error for P2(x-t,y)");
        check(sum.max_conservation_residual < 1.0e-12, "Unstructured ENO3 conservation P2");
    }

    // 2. Smooth bump EOC with ENO2 (SSP_RK2) and ENO3 (SSP_RK3)
    auto smooth_ic = [](Real x, Real y) -> Real {
        const Real dx = x - 0.75;
        const Real dy = y - 0.5;
        const Real r = std::hypot(dx, dy);
        const Real R = 0.42;
        if (r >= R) return 0.0;
        const Real c = std::cos(0.5 * PI * r / R);
        const Real c2 = c * c;
        return c2 * c2 * c2;  // C^5 bump for clean 3rd-order testing
    };

    auto eval_smooth_tri = [&](const UnstructuredMesh2D& m, int order) -> Real {
        const Real T = 0.5;
        std::vector<Real> u, u_ex;
        m.initialize_cell_averages(u, [&](Real x, Real y) { return smooth_ic(x, y); }, 5);
        m.initialize_cell_averages(u_ex, [&](Real x, Real y) { return smooth_ic(x - T, y); }, 5);
        solvers::UnstructuredAdvectionSolver solver(
            m, velocity, [&](Real x, Real y, Real t, int) { return smooth_ic(x - t, y); },
            solvers::ReconstructionScheme::ENO, 3,
            solvers::UnstructuredLimiter::BarthJespersen, order);
        solvers::UnstructuredRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::ENO;
        opts.eno_order = order;
        opts.rk_method = (order == 2) ? time::RKMethod::SSP_RK2 : time::RKMethod::SSP_RK3;
        opts.cfl = 0.4;
        opts.t_end = T;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-12, "Unstructured ENO smooth conservation");
        return compute_error_norms(u, u_ex, m.cell_areas()).l1;
    };

    const Real e2_f = eval_smooth_tri(fine, 2);
    const Real e2_f2 = eval_smooth_tri(fine2, 2);
    const Real eoc2 = experimental_order(e2_f, e2_f2, fine.h_max_diam(), fine2.h_max_diam());

    const Real e3_f = eval_smooth_tri(fine, 3);
    const Real e3_f2 = eval_smooth_tri(fine2, 3);
    const Real eoc3 = experimental_order(e3_f, e3_f2, fine.h_max_diam(), fine2.h_max_diam());

    check(eoc2 > 1.2, "Unstructured ENO2 smooth C^5 EOC_L1 > 1.2");
    check(e3_f2 < 0.4 * e2_f2, "Unstructured ENO3 much more accurate than ENO2 on fine2");
    check(eoc3 > 2.2, "Unstructured ENO3 smooth C^5 EOC_L1 > 2.2");

    std::cout << "[PASS] unstructured_eno (ENO2 EOC=" << eoc2
              << ", ENO3 E1=" << e3_f2 << ", ENO3 EOC=" << eoc3 << ")\n";
}

void test_kpp_eno() {
    const Real u_in = 3.5 * PI;
    const Real u_out = 0.25 * PI;
    StructuredMesh2D mesh(-2.5, 2.5, -2.5, 2.5, 100, 100, 3);
    auto u = mesh.create_field();
    const Real dx = mesh.dx(), dy = mesh.dy(), area = mesh.cell_area();
    for (Index j = 0; j < mesh.ny(); ++j) {
        const Real y0 = mesh.ymin() + static_cast<Real>(j) * dy;
        for (Index i = 0; i < mesh.nx(); ++i) {
            const Real x0 = mesh.xmin() + static_cast<Real>(i) * dx;
            const Real frac = geom::disk_intersection_area_rect(x0, x0 + dx, y0, y0 + dy, 1.0) / area;
            u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = u_out + (u_in - u_out) * frac;
        }
    }

    solvers::StructuredConservationLaw law;
    law.fx = [](Real, Real, Real v) { return std::sin(v); };
    law.fy = [](Real, Real, Real v) { return std::cos(v); };
    law.alpha_x = [](Real, Real, Real, Real) { return 1.0; };
    law.alpha_y = [](Real, Real, Real, Real) { return 1.0; };
    law.max_speeds = [](const StructuredField2D&, Real& ax, Real& ay) { ax = 1.0; ay = 1.0; };

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletExact;
    bc.left.value = [u_out](Real, Real, Real) { return u_out; };
    bc.right = bc.left;
    bc.bottom = bc.left;
    bc.top = bc.left;

    solvers::StructuredFVSolver solver(
        mesh, bc, law, solvers::ReconstructionScheme::ENO, solvers::TVDSlopeLimiter::MC, 3);
    solvers::SolverRunOptions opts;
    opts.scheme = solvers::ReconstructionScheme::ENO;
    opts.eno_order = 3;
    opts.rk_method = time::RKMethod::SSP_RK3;
    opts.cfl = 0.45;
    opts.t_end = 1.0;
    const auto sum = solver.run(u, opts);
    check(sum.max_conservation_residual < 1.0e-12, "KPP ENO3 conservation residual R_M^n ~ 0");
    std::cout << "[PASS] kpp_eno (u in [" << sum.final_stats.min_val << ", "
              << sum.final_stats.max_val << "], max_RM=" << sum.max_conservation_residual << ")\n";
}

void test_structured_weno_and_eoc() {
    // 1. Aligned stationary band stays machine-exact under WENO5 + SSP_RK3
    {
        StructuredMesh2D mesh(0.0, 1.0, 0.0, 1.0, 40, 40, 3);
        auto u = mesh.create_field();
        auto u_exact = mesh.create_field();
        for (Index j = 0; j < mesh.ny(); ++j) {
            const Real y0 = static_cast<Real>(j) * mesh.dy();
            const Real y1 = y0 + mesh.dy();
            const Real frac = geom::interval_overlap_fraction(y0, y1, 0.3, 0.7);
            for (Index i = 0; i < mesh.nx(); ++i) {
                u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = frac;
                u_exact(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = frac;
            }
        }
        StructuredBoundaryConfig bc;
        bc.left.type = StructuredBCType::DirichletFaceAvg;
        bc.left.face_avg = [](Real y0, Real y1, Real) {
            return geom::interval_overlap_fraction(y0, y1, 0.3, 0.7);
        };
        bc.right.type = StructuredBCType::ZeroGradient;
        bc.bottom.type = StructuredBCType::ZeroGradient;
        bc.top.type = StructuredBCType::ZeroGradient;

        solvers::StructuredFVSolver solver(
            mesh, bc, solvers::make_linear_advection_law(1.0, 0.0),
            solvers::ReconstructionScheme::WENO, solvers::TVDSlopeLimiter::MC,
            3, solvers::WENOWeightType::WENO_Z);
        solvers::SolverRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::WENO;
        opts.weno_type = solvers::WENOWeightType::WENO_Z;
        opts.rk_method = time::RKMethod::SSP_RK3;
        opts.cfl = 0.45;
        opts.t_end = 1.0;
        const auto sum = solver.run(u, opts);
        const auto err = compute_error_norms(u, u_exact, mesh.cell_area());
        check(err.l1 < 1.0e-13, "WENO5 aligned control exact E1 < 1e-13");
        check(sum.max_conservation_residual < 1.0e-13, "WENO5 aligned conservation R_M^n ~ 0");
    }

    // 2. Smooth periodic diagonal sine wave achieves 5th-order EOC under WENO5-Z
    auto smooth_periodic_fn = [](Real x, Real y) -> Real {
        return 0.5 + 0.5 * std::sin(2.0 * PI * (y - x));
    };
    const auto bc_per = StructuredBoundaryConfig::all_periodic();
    const auto law_diag = solvers::make_linear_advection_law(1.0, 1.0);

    auto eval_periodic = [&](Index n, solvers::WENOWeightType wtype) -> ErrorNorms {
        StructuredMesh2D mesh(0.0, 1.0, 0.0, 1.0, n, n, 3);
        auto u = mesh.create_field();
        auto u_exact = mesh.create_field();
        mesh.initialize_cell_averages(u, smooth_periodic_fn, 5);
        u_exact = u;
        solvers::StructuredFVSolver solver(
            mesh, bc_per, law_diag, solvers::ReconstructionScheme::WENO,
            solvers::TVDSlopeLimiter::MC, 3, wtype);
        solvers::SolverRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::WENO;
        opts.weno_type = wtype;
        opts.rk_method = time::RKMethod::SSP_RK3;
        opts.cfl = 0.45;
        opts.t_end = 1.0;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-12, "Structured WENO conservation ~ 0");
        return compute_error_norms(u, u_exact, mesh.cell_area());
    };

    const auto ez20 = eval_periodic(20, solvers::WENOWeightType::WENO_Z);
    const auto ez40 = eval_periodic(40, solvers::WENOWeightType::WENO_Z);
    const Real eoc_z1 = experimental_order(ez20.l1, ez40.l1, 1.0 / 20.0, 1.0 / 40.0);
    const Real eoc_zinf = experimental_order(ez20.linf, ez40.linf, 1.0 / 20.0, 1.0 / 40.0);
    check(eoc_z1 > 4.6, "WENO5-Z smooth periodic EOC_L1 > 4.6");
    check(eoc_zinf > 4.5, "WENO5-Z smooth periodic EOC_Linf > 4.5");

    // 3. Nonlinear 2D KPP smooth short-time test (3-pt Gauss WENO5 achieves >4th/5th-order!)
    auto smooth_nl_ic = [](Real x, Real y) -> Real {
        return 1.0 + 0.25 * std::sin(2.0 * PI * x) * std::sin(2.0 * PI * y);
    };
    solvers::StructuredConservationLaw nl_law;
    nl_law.fx = [](Real, Real, Real v) { return std::sin(v); };
    nl_law.fy = [](Real, Real, Real v) { return std::cos(v); };
    nl_law.alpha_x = [](Real, Real, Real, Real) { return 1.0; };
    nl_law.alpha_y = [](Real, Real, Real, Real) { return 1.0; };
    nl_law.max_speeds = [](const StructuredField2D&, Real& ax, Real& ay) { ax = 1.0; ay = 1.0; };

    auto run_nl = [&](Index n) -> StructuredField2D {
        StructuredMesh2D m(0.0, 1.0, 0.0, 1.0, n, n, 3);
        auto f = m.create_field();
        m.initialize_cell_averages(f, smooth_nl_ic, 5);
        solvers::StructuredFVSolver s(
            m, bc_per, nl_law, solvers::ReconstructionScheme::WENO,
            solvers::TVDSlopeLimiter::MC, 3, solvers::WENOWeightType::WENO_Z);
        solvers::SolverRunOptions o;
        o.scheme = solvers::ReconstructionScheme::WENO;
        o.rk_method = time::RKMethod::RK4;
        o.cfl = 0.25;
        o.t_end = 0.04;
        s.run(f, o);
        return f;
    };
    auto avg_down = [](const StructuredField2D& fine, Index nc) {
        const Index rx = fine.nx() / nc;
        StructuredField2D c(nc, nc, 0, 0.0);
        const Real inv = 1.0 / static_cast<Real>(rx * rx);
        for (Index j = 0; j < nc; ++j) {
            for (Index i = 0; i < nc; ++i) {
                Real s = 0.0;
                for (Index sj = 0; sj < rx; ++sj) {
                    for (Index si = 0; si < rx; ++si) {
                        s += fine(static_cast<SignedIndex>(i * rx + si),
                                  static_cast<SignedIndex>(j * rx + sj));
                    }
                }
                c(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = s * inv;
            }
        }
        return c;
    };

    const auto ref_nl = run_nl(160);
    const auto nl20 = run_nl(20);
    const auto nl40 = run_nl(40);
    const Real e_nl20 = compute_error_norms(nl20, avg_down(ref_nl, 20), 1.0 / 400.0).l1;
    const Real e_nl40 = compute_error_norms(nl40, avg_down(ref_nl, 40), 1.0 / 1600.0).l1;
    const Real eoc_nl = experimental_order(e_nl20, e_nl40, 1.0 / 20.0, 1.0 / 40.0);
    check(eoc_nl > 4.4, "2D nonlinear WENO5-Z with 3-pt face Gauss achieves >4.4th order");

    std::cout << "[PASS] structured_weno_and_eoc (WENO5-Z EOC_1=" << eoc_z1
              << ", Linf_EOC=" << eoc_zinf << ", nonlinear_2D_EOC=" << eoc_nl << ")\n";
}

void test_unstructured_weno() {
    const UnstructuredMesh2D base = generate_unstructured_tri_mesh(
        0.0, 4.0, 0.0, 1.0, 16, 4, 0.35, 5, 42);
    const UnstructuredMesh2D fine = base.refine_uniform();
    const UnstructuredMesh2D fine2 = fine.refine_uniform();
    auto velocity = [](Real, Real) -> Vec2 { return {1.0, 0.0}; };

    // 1. Free-stream preservation u == 1 and exact quadratic P2 spatial RHS
    {
        std::vector<Real> u(base.num_cells(), 1.0);
        solvers::UnstructuredAdvectionSolver solver(
            base, velocity, [](Real, Real, Real, int) { return 1.0; },
            solvers::ReconstructionScheme::WENO, 3, solvers::UnstructuredLimiter::BarthJespersen,
            3, solvers::WENOWeightType::WENO_Z);
        solvers::UnstructuredRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::WENO;
        opts.rk_method = time::RKMethod::SSP_RK3;
        opts.cfl = 0.45;
        opts.t_end = 0.5;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-13, "Unstructured WENO freestream R_M");
        for (Real val : u) {
            check(std::abs(val - 1.0) < 1.0e-12, "Unstructured WENO freestream u == 1");
        }
    }
    {
        auto p2_fn = [](Real x, Real y) -> Real {
            return 1.0 + 0.2 * x - 0.3 * y + 0.1 * x * x - 0.15 * x * y + 0.08 * y * y;
        };
        std::vector<Real> u, dudt, dudt_ex;
        base.initialize_cell_averages(u, p2_fn, 5);
        solvers::UnstructuredAdvectionSolver solver(
            base, velocity, [&](Real x, Real y, Real t, int) { return p2_fn(x - t, y); },
            solvers::ReconstructionScheme::WENO, 3, solvers::UnstructuredLimiter::BarthJespersen,
            3, solvers::WENOWeightType::WENO_Z);
        solver.compute_rhs(u, 0.0, dudt);
        base.initialize_cell_averages(
            dudt_ex, [](Real x, Real y) { return -(0.2 + 0.2 * x - 0.15 * y); }, 5);
        Real max_rhs_err = 0.0;
        for (Index c = 0; c < base.num_cells(); ++c) {
            max_rhs_err = std::max(max_rhs_err, std::abs(dudt[c] - dudt_ex[c]));
        }
        check(max_rhs_err < 1.0e-11, "Unstructured WENO exact spatial RHS for P2");
    }

    // 2. Smooth C^5 bump EOC on unstructured triangles with WENO-Z + SSP_RK3
    auto smooth_ic = [](Real x, Real y) -> Real {
        const Real dx = x - 0.75;
        const Real dy = y - 0.5;
        const Real r = std::hypot(dx, dy);
        const Real R = 0.42;
        if (r >= R) return 0.0;
        const Real c = std::cos(0.5 * PI * r / R);
        const Real c2 = c * c;
        return c2 * c2 * c2;
    };
    auto broad_ic = [](Real x, Real y) -> Real {
        return std::sin(PI * x) * std::sin(PI * y);
    };
    auto eval_tri = [&](const UnstructuredMesh2D& m, const auto& ic) -> ErrorNorms {
        const Real T = 0.5;
        std::vector<Real> u, u_ex;
        m.initialize_cell_averages(u, [&](Real x, Real y) { return ic(x, y); }, 5);
        m.initialize_cell_averages(u_ex, [&](Real x, Real y) { return ic(x - T, y); }, 5);
        solvers::UnstructuredAdvectionSolver solver(
            m, velocity, [&](Real x, Real y, Real t, int) { return ic(x - t, y); },
            solvers::ReconstructionScheme::WENO, 3, solvers::UnstructuredLimiter::BarthJespersen,
            3, solvers::WENOWeightType::WENO_Z);
        solvers::UnstructuredRunOptions opts;
        opts.scheme = solvers::ReconstructionScheme::WENO;
        opts.rk_method = time::RKMethod::SSP_RK3;
        opts.cfl = 0.4;
        opts.t_end = T;
        const auto sum = solver.run(u, opts);
        check(sum.max_conservation_residual < 1.0e-12, "Unstructured WENO smooth conservation");
        return compute_error_norms(u, u_ex, m.cell_areas());
    };

    const ErrorNorms err_c_f = eval_tri(fine, smooth_ic);
    const ErrorNorms err_c_f2 = eval_tri(fine2, smooth_ic);
    const Real eoc_cos6 = experimental_order(err_c_f.l1, err_c_f2.l1, fine.h_max_diam(), fine2.h_max_diam());
    check(eoc_cos6 > 2.3, "Unstructured WENO cos6 EOC_L1 > 2.3");

    const ErrorNorms err_b_f = eval_tri(fine, broad_ic);
    const ErrorNorms err_b_f2 = eval_tri(fine2, broad_ic);
    const Real eoc_b1 = experimental_order(err_b_f.l1, err_b_f2.l1, fine.h_max_diam(), fine2.h_max_diam());
    const Real eoc_binf = experimental_order(err_b_f.linf, err_b_f2.linf, fine.h_max_diam(), fine2.h_max_diam());
    check(eoc_b1 > 2.85, "Unstructured WENO resolved C^inf wave EOC_L1 > 2.85");
    check(eoc_binf > 2.65, "Unstructured WENO resolved C^inf wave EOC_Linf > 2.65");

    std::cout << "[PASS] unstructured_weno (cos6 EOC_L1=" << eoc_cos6
              << ", sin EOC_L1=" << eoc_b1 << ", sin EOC_Linf=" << eoc_binf << ")\n";
}

void test_kpp_weno() {
    const Real u_in = 3.5 * PI;
    const Real u_out = 0.25 * PI;
    StructuredMesh2D mesh(-2.5, 2.5, -2.5, 2.5, 100, 100, 3);
    auto u = mesh.create_field();
    const Real dx = mesh.dx(), dy = mesh.dy(), area = mesh.cell_area();
    for (Index j = 0; j < mesh.ny(); ++j) {
        const Real y0 = mesh.ymin() + static_cast<Real>(j) * dy;
        for (Index i = 0; i < mesh.nx(); ++i) {
            const Real x0 = mesh.xmin() + static_cast<Real>(i) * dx;
            const Real frac = geom::disk_intersection_area_rect(x0, x0 + dx, y0, y0 + dy, 1.0) / area;
            u(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) = u_out + (u_in - u_out) * frac;
        }
    }

    solvers::StructuredConservationLaw law;
    law.fx = [](Real, Real, Real v) { return std::sin(v); };
    law.fy = [](Real, Real, Real v) { return std::cos(v); };
    law.alpha_x = [](Real, Real, Real, Real) { return 1.0; };
    law.alpha_y = [](Real, Real, Real, Real) { return 1.0; };
    law.max_speeds = [](const StructuredField2D&, Real& ax, Real& ay) { ax = 1.0; ay = 1.0; };

    StructuredBoundaryConfig bc;
    bc.left.type = StructuredBCType::DirichletExact;
    bc.left.value = [u_out](Real, Real, Real) { return u_out; };
    bc.right = bc.left;
    bc.bottom = bc.left;
    bc.top = bc.left;

    solvers::StructuredFVSolver solver(
        mesh, bc, law, solvers::ReconstructionScheme::WENO, solvers::TVDSlopeLimiter::MC,
        3, solvers::WENOWeightType::WENO_Z);
    solvers::SolverRunOptions opts;
    opts.scheme = solvers::ReconstructionScheme::WENO;
    opts.weno_type = solvers::WENOWeightType::WENO_Z;
    opts.rk_method = time::RKMethod::SSP_RK3;
    opts.cfl = 0.45;
    opts.t_end = 1.0;
    const auto sum = solver.run(u, opts);
    check(sum.max_conservation_residual < 1.0e-12, "KPP WENO conservation residual R_M^n ~ 0");
    std::cout << "[PASS] kpp_weno (u in [" << sum.final_stats.min_val << ", "
              << sum.final_stats.max_val << "], max_RM=" << sum.max_conservation_residual << ")\n";
}

void test_mpi_parallel_equivalence(const mpi::MpiGuard& mpi_guard) {
    const bool is_root = (mpi_guard.rank() == 0);

    // 1. Structured WENO5-Z periodic (exercises wrap-around halo exchange & corners)
    {
        const Index N = 40;
        auto smooth_per = [](Real x, Real y) {
            return 0.5 + 0.5 * std::sin(2.0 * PI * (y - x));
        };
        const auto bc_per = StructuredBoundaryConfig::all_periodic();
        const auto law_diag = solvers::make_linear_advection_law(1.0, 1.0);

        StructuredField2D ref_serial;
        solvers::SolverRunSummary ref_sum;
        if (is_root) {
            StructuredMesh2D gmesh(0.0, 1.0, 0.0, 1.0, N, N, 3);
            ref_serial = gmesh.create_field();
            gmesh.initialize_cell_averages(ref_serial, smooth_per, 5);
            solvers::StructuredFVSolver s_ser(
                gmesh, bc_per, law_diag, solvers::ReconstructionScheme::WENO,
                solvers::TVDSlopeLimiter::MC, 3, solvers::WENOWeightType::WENO_Z);
            solvers::SolverRunOptions o;
            o.scheme = solvers::ReconstructionScheme::WENO;
            o.weno_type = solvers::WENOWeightType::WENO_Z;
            o.rk_method = time::RKMethod::SSP_RK3;
            o.cfl = 0.45;
            o.t_end = 0.25;
            ref_sum = s_ser.run(ref_serial, o);
        }

        const auto decomp = mpi::CartDecomp2D::create(N, N, true, true);
        StructuredMesh2D lmesh = decomp.make_local_mesh(0.0, 1.0, 0.0, 1.0, 3);
        auto u_loc = lmesh.create_field();
        lmesh.initialize_cell_averages(u_loc, smooth_per, 5);
        solvers::StructuredFVSolver s_par(
            lmesh, bc_per, law_diag, solvers::ReconstructionScheme::WENO,
            solvers::TVDSlopeLimiter::MC, 3, solvers::WENOWeightType::WENO_Z, &decomp);
        solvers::SolverRunOptions o;
        o.scheme = solvers::ReconstructionScheme::WENO;
        o.weno_type = solvers::WENOWeightType::WENO_Z;
        o.rk_method = time::RKMethod::SSP_RK3;
        o.cfl = 0.45;
        o.t_end = 0.25;
        const auto par_sum = s_par.run(u_loc, o);
        const auto gathered = mpi::gather_structured_field(u_loc, decomp);
        if (is_root) {
            Real max_diff = 0.0;
            for (Index j = 0; j < N; ++j) {
                for (Index i = 0; i < N; ++i) {
                    const SignedIndex is = static_cast<SignedIndex>(i);
                    const SignedIndex js = static_cast<SignedIndex>(j);
                    max_diff = std::max(max_diff, std::abs(gathered(is, js) - ref_serial(is, js)));
                }
            }
            check(max_diff < 1.0e-14, "MPI structured periodic WENO5-Z matches serial pointwise");
            check(std::abs(par_sum.final_stats.total_mass - ref_sum.final_stats.total_mass) < 1.0e-14,
                  "MPI structured periodic mass matches serial");
        }
    }

    // 2. Structured WENO5-Z non-periodic (DirichletCellAvg + Extrapolate + interior MPI corners)
    {
        const Index N = 40;
        auto smooth_band = [](Real x, Real y) {
            const Real d = (y - x) / 0.2;
            return std::exp(-d * d);
        };
        StructuredBoundaryConfig bc;
        bc.left.type = StructuredBCType::DirichletCellAvg;
        bc.left.value = [&](Real x, Real y, Real) { return smooth_band(x, y); };
        bc.bottom = bc.left;
        bc.right.type = StructuredBCType::Extrapolate;
        bc.top.type = StructuredBCType::Extrapolate;
        const auto law_diag = solvers::make_linear_advection_law(1.0, 1.0);

        StructuredField2D ref_serial;
        if (is_root) {
            StructuredMesh2D gmesh(0.0, 1.0, 0.0, 1.0, N, N, 3);
            ref_serial = gmesh.create_field();
            gmesh.initialize_cell_averages(ref_serial, smooth_band, 4);
            solvers::StructuredFVSolver s_ser(
                gmesh, bc, law_diag, solvers::ReconstructionScheme::WENO,
                solvers::TVDSlopeLimiter::MC, 3, solvers::WENOWeightType::WENO_Z);
            solvers::SolverRunOptions o;
            o.scheme = solvers::ReconstructionScheme::WENO;
            o.weno_type = solvers::WENOWeightType::WENO_Z;
            o.rk_method = time::RKMethod::SSP_RK3;
            o.cfl = 0.45;
            o.t_end = 0.25;
            s_ser.run(ref_serial, o);
        }

        const auto decomp = mpi::CartDecomp2D::create(N, N, false, false);
        StructuredMesh2D lmesh = decomp.make_local_mesh(0.0, 1.0, 0.0, 1.0, 3);
        auto u_loc = lmesh.create_field();
        lmesh.initialize_cell_averages(u_loc, smooth_band, 4);
        solvers::StructuredFVSolver s_par(
            lmesh, bc, law_diag, solvers::ReconstructionScheme::WENO,
            solvers::TVDSlopeLimiter::MC, 3, solvers::WENOWeightType::WENO_Z, &decomp);
        solvers::SolverRunOptions o;
        o.scheme = solvers::ReconstructionScheme::WENO;
        o.weno_type = solvers::WENOWeightType::WENO_Z;
        o.rk_method = time::RKMethod::SSP_RK3;
        o.cfl = 0.45;
        o.t_end = 0.25;
        const auto par_sum = s_par.run(u_loc, o);
        check(par_sum.max_conservation_residual < 1.0e-13, "MPI structured non-periodic conservation");
        const auto gathered = mpi::gather_structured_field(u_loc, decomp);
        if (is_root) {
            Real max_diff = 0.0;
            for (Index j = 0; j < N; ++j) {
                for (Index i = 0; i < N; ++i) {
                    const SignedIndex is = static_cast<SignedIndex>(i);
                    const SignedIndex js = static_cast<SignedIndex>(j);
                    max_diff = std::max(max_diff, std::abs(gathered(is, js) - ref_serial(is, js)));
                }
            }
            check(max_diff < 1.0e-14, "MPI structured non-periodic WENO5-Z matches serial pointwise");
        }
    }

    // 3. Unstructured Venkatakrishnan TVD and WENO-Z P2 MPI vs serial pointwise equivalence
    {
        const UnstructuredMesh2D base = generate_unstructured_tri_mesh(
            0.0, 4.0, 0.0, 1.0, 16, 4, 0.35, 5, 42);
        const UnstructuredMesh2D mesh = base.refine_uniform();
        auto velocity = [](Real, Real) -> Vec2 { return {1.0, 0.0}; };
        auto ic_fn = [](Real x, Real y) { return std::sin(PI * x) * std::sin(PI * y); };
        auto inflow_fn = [&](Real x, Real y, Real t, int) { return ic_fn(x - t, y); };

        std::vector<Real> u0;
        mesh.initialize_cell_averages(u0, ic_fn, 5);
        const auto u_decomp = mpi::UnstructuredDecomp::create(mesh);

        auto verify_unstruct = [&](solvers::ReconstructionScheme sch, time::RKMethod rk,
                                   solvers::UnstructuredLimiter lim, const char* label) {
            std::vector<Real> u_ser = u0;
            if (is_root) {
                solvers::UnstructuredAdvectionSolver s_ser(
                    mesh, velocity, inflow_fn, sch, 3, lim, 3, solvers::WENOWeightType::WENO_Z);
                solvers::UnstructuredRunOptions opts;
                opts.scheme = sch;
                opts.limiter = lim;
                opts.rk_method = rk;
                opts.cfl = 0.4;
                opts.t_end = 0.25;
                s_ser.run(u_ser, opts);
            }
            std::vector<Real> u_par = u0;
            solvers::UnstructuredAdvectionSolver s_par(
                mesh, velocity, inflow_fn, sch, 3, lim, 3, solvers::WENOWeightType::WENO_Z, &u_decomp);
            solvers::UnstructuredRunOptions opts;
            opts.scheme = sch;
            opts.limiter = lim;
            opts.rk_method = rk;
            opts.cfl = 0.4;
            opts.t_end = 0.25;
            const auto sum_par = s_par.run(u_par, opts);
            check(sum_par.max_conservation_residual < 1.0e-12, "MPI unstructured conservation");
            if (is_root) {
                Real max_diff = 0.0;
                for (Index c = 0; c < mesh.num_cells(); ++c) {
                    max_diff = std::max(max_diff, std::abs(u_par[c] - u_ser[c]));
                }
                check(max_diff < 1.0e-14, label);
            }
        };

        verify_unstruct(solvers::ReconstructionScheme::TVD, time::RKMethod::SSP_RK2,
                        solvers::UnstructuredLimiter::Venkatakrishnan,
                        "MPI unstructured Venkatakrishnan TVD matches serial pointwise");
        verify_unstruct(solvers::ReconstructionScheme::WENO, time::RKMethod::SSP_RK3,
                        solvers::UnstructuredLimiter::BarthJespersen,
                        "MPI unstructured WENO-Z P2 matches serial pointwise");
    }

    if (is_root) {
        std::cout << "[PASS] mpi_parallel_equivalence (ranks=" << mpi_guard.size() << ")\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    mpi::MpiGuard mpi_guard(argc, argv);
    if (mpi_guard.rank() == 0) {
        test_diagonal_band_and_aligned_control();
        test_unstructured_tube_conservation_and_bounds();
        test_kpp_pdf4_bounds_and_conservation();
        test_structured_tvd_and_eoc();
        test_unstructured_tvd();
        test_kpp_tvd();
        test_structured_eno_and_eoc();
        test_unstructured_eno();
        test_kpp_eno();
        test_structured_weno_and_eoc();
        test_unstructured_weno();
        test_kpp_weno();
    }
    test_mpi_parallel_equivalence(mpi_guard);
    if (mpi_guard.rank() == 0) {
        std::cout << "All benchmark solver tests passed.\n";
    }
    return 0;
}
