# 2D Finite Volume Method (FVM) Solver Suite

A self-contained C++20 library and benchmark suite for **2D scalar conservation laws** on structured Cartesian grids and unstructured triangular meshes. With zero external PDE or linear-algebra dependencies in the serial build, it compares four finite-volume reconstruction families—**1st-order monotone**, **2nd-order TVD**, **3rd-order ENO**, and **WENO**—with optional OpenMPI domain decomposition and native ParaView XML output.

---

## Features

- **Four Reconstruction Families with SSP Time Integration:**
  - **Structured Cartesian grids (`StructuredFVSolver`):**
    - `1st-Order` upwind / global Rusanov with Forward Euler
    - `TVD` piecewise-linear MUSCL (`minmod`, `mc`, `vanleer`, `superbee`) with `SSP-RK2`
    - `ENO3` Newton divided-difference reconstruction with 2-point Gauss face quadrature and `SSP-RK3` (`ENO2` + `SSP-RK2` also available in the library)
    - `WENO5` (WENO-Z and WENO-JS) with 3-point Gauss face quadrature and Shi–Hu–Shu negative-weight splitting (`SSP-RK3`)
  - **Unstructured triangular meshes (`UnstructuredAdvectionSolver`):**
    - `1st-Order` piecewise-constant upwind with Forward Euler
    - `TVD` distance-weighted least-squares linear reconstruction (`Barth–Jespersen` and `Venkatakrishnan` limiters) with `SSP-RK2`
    - `ENO3` $P^2$ quadratic polynomial reconstruction over central + directional $60^\circ$ sector stencils with `SSP-RK3`
    - `WENO` $P^2$ weighted combination of central and directional sector polynomials (`WENO_Z` / `WENO_JS`) with `SSP-RK3`
- **Self-Contained Numerics & Geometry:**
  - Dense LU and rank/condition-checked Householder QR (`include/fvm/linalg/dense.hpp`)
  - CSR sparse matrices with Jacobi/SGS-preconditioned CG and BiCGSTAB (`include/fvm/linalg/sparse.hpp`)
  - Exact geometric cell-average initialization via polygon half-plane clipping and disk–rectangle overlap (`include/fvm/mesh/geometry_exact.hpp`)
  - Built-in Bowyer–Watson Delaunay triangle generator with Lloyd smoothing, 1-to-4 uniform refinement, and Gmsh v2.2 reader (`include/fvm/mesh/`)
  - Butcher-weight-aware discrete conservation residual $R_M^n$ monitored per time step
- **Optional OpenMPI Parallelism:** Compile-time MPI support (`USE_MPI=1` / `-DFVM_USE_MPI=ON`) with 2D Cartesian halo exchanges (including diagonal corners) and centroid-partitioned unstructured halo schedules, matching serial solutions to roundoff precision.
- **ParaView & CSV Output:** Native `.vti`, `.vtu`, and `.pvd` time series alongside CSV convergence tables and 1D profile slices.

---

## Requirements & Building

### Prerequisites
- C++20 compiler (`clang++` or `g++`)
- GNU `make` or `cmake` ($\ge 3.16$)
- **Optional:** MPI (`mpicxx` and `mpiexec`, e.g. OpenMPI) for distributed execution; [ParaView](https://www.paraview.org/) for visualizing `.pvd`/`.vti`/`.vtu` outputs

### Build with `make`

```bash
# Serial build and unit/verification tests
make all
make test

# OpenMPI build into build_mpi/ and 4-rank tests (override rank count with NP=<n>)
make USE_MPI=1 all
make USE_MPI=1 test
```

### Build with `cmake`

```bash
# Serial
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure

# With MPI enabled
cmake -B build_mpi -DFVM_USE_MPI=ON
cmake --build build_mpi
mpiexec -n 4 ./build_mpi/test_core
mpiexec -n 4 ./build_mpi/test_solvers
```

---

## Solver Overview

| Family | Structured Grid | Unstructured Triangles | Time Integrator |
|---|---|---|---|
| `1st-Order` | Piecewise constant, 1-pt face flux | Piecewise constant, 1-pt face flux | Forward Euler |
| `TVD` | Dimension-by-dimension limited slope | Least-squares $\nabla u_c$ (Householder QR) + vertex limiter | `SSP-RK2` |
| `ENO3` | 3-cell Newton divided-difference + 2-pt Gauss face quadrature | Central + $60^\circ$ directional $P^2$ stencils + 3-pt Gauss edge quadrature | `SSP-RK3` |
| `WENO` | 5th-order WENO-Z/JS + 3-pt Gauss face quadrature (Shi–Hu–Shu splitting) | Convexly weighted $P^2$ central + sector polynomials (`WENO_Z` / `WENO_JS`) | `SSP-RK3` |

Smooth formal convergence orders (2nd-order `TVD`, 3rd-order `ENO3`, 5th-order structured `WENO5-Z`, and 3rd-order unstructured $P^2$ `WENO-Z`) are verified on linear and nonlinear smooth test cases in [tests/test_solvers.cpp](tests/test_solvers.cpp). The benchmark drivers focus on sharp discontinuities and nonconvex wave structures where diffusion, limiters, mesh geometry, and entropy effects compete.

---

## Benchmark Problems

### 1. Diagonal Band Advection (`problem1`)
- **Equation & Domain:** $u_t + u_x + u_y = 0$ on $(0,1)^2$, $T = 1.0$, $\mathrm{CFL} = 0.45$, grids $N \in \{40, 80, 160, 320\}$.
- **Setup:** Stationary $45^\circ$ contact band $u_0(x,y) = \mathbf{1}_{\{|y-x| < 0.2\}}$ initialized via exact cell-area intersection, exact inflow face averages on $x=0$ and $y=0$, and zero-gradient outflow on $x=1, y=1$.
- **Variants:** `1st-Order`, `TVD-mc`, `ENO3`, `WENO5-z`.
- **What it tests:** Cross-wind numerical viscosity on non-grid-aligned discontinuities. Asymptotically across a linear contact discontinuity, a $p$-th order scheme scales in $L^1$ as $h^{p/(p+1)}$ ($0.50$, $0.67$, $0.83$). At $320^2$, `WENO5-z` narrows the 10–90% front width to `0.0119` (vs. `0.0907` for `1st-Order`), while `TVD-mc` stays strictly within $[0, 1]$ (`max_RM <= 1.9e-14` across all runs).

<details>
<summary>Representative convergence metrics on 40² → 320²</summary>

| Scheme | $L^1$ ($320^2$) | `EOC_1` | $L^2$ ($320^2$) | $d_{10\text{--}90}(x+y=1)$ | Bounds $[u_{\min}, u_{\max}]$ |
|---|---:|---:|---:|---:|---|
| `1st-Order` | `5.78e-02` | `0.47` | `1.28e-01` | `0.0907` | `[8.6e-78, 1]` |
| `TVD-mc` | `8.38e-03` | `0.68` | `4.69e-02` | `0.0170` | `[0, 1]` |
| `ENO3` | `1.06e-02` | `0.67` | `4.91e-02` | `0.0200` | `[1.1e-170, 1]` |
| `WENO5-z` | `5.34e-03` | `0.78` | `3.27e-02` | `0.0119` | `[-1.5e-06, 1]` |

</details>

### 2. Unstructured Tube Transport (`problem2`)
- **Equation & Domain:** $u_t + u_x = 0$ on $\Omega = (0,4)\times(0,1)$, $T = 2.0$, $\mathrm{CFL} = 0.45$, zero inflow at $x=0$.
- **Setup:** Discontinuous square patch $S_0 = [0.5, 1.0]\times[0.25, 0.75]$ advected to $S(T) = [2.5, 3.0]\times[0.25, 0.75]$. Evaluated on four unstructured levels (`264` to `16,896` triangles) under two mesh-generation modes:
  - `--mesh nested` (default): Lloyd-smoothed Delaunay base mesh refined self-similarly 1-to-4 ($q_{\min} = 0.815$).
  - `--mesh irregular`: independent unsmoothed Delaunay triangulation at every refinement level ($q_{\min} = 0.699 \to 0.566$).
- **Variants:** `1st-Tri`, `TVD-Tri` (`Barth–Jespersen`), `ENO3-Tri`, `WENO-Tri` ($P^2$ `WENO_Z`).
- **What it tests:** Cross-stream leakage (mass escaping $y \in [0.25, 0.75]$) vs. streamwise spill under triangle orientation bias and mesh irregularity. Without smoothing or nested refinement (`--mesh irregular`), cross-stream leakage increases substantially (`5.8x` for `1st-Tri`, `2.65x` for `WENO-Tri` at `16,896` cells), yet Householder QR stencil checks keep $P^2$ `ENO3` and `WENO` well-conditioned and mass-conservative (`max_RM <= 2.2e-15`).

<details>
<summary>Representative patch results at finest level (ref = 3, 16,896 triangles)</summary>

| Mesh Mode | Scheme | Tri $L^1$ | `EOC_1` | Cross leak | Stream spill | Bounds $[u_{\min}, u_{\max}]$ |
|---|---|---:|---:|---:|---:|---|
| `nested` | `1st-Tri` | `1.03e-01` | `0.44` | `4.90e-03` | `4.96e-02` | `[0, 0.952]` |
| `nested` | `TVD-Tri` | `3.42e-02` | `0.69` | `3.71e-03` | `1.65e-02` | `[-5.9e-14, 1]` |
| `nested` | `ENO3-Tri` | `3.11e-02` | `0.69` | `3.66e-03` | `1.50e-02` | `[-0.0029, 1.0089]` |
| `nested` | `WENO-Tri` | `3.02e-02` | `0.67` | `3.70e-03` | `1.46e-02` | `[-0.0038, 1.0040]` |
| `irregular` | `1st-Tri` | `1.49e-01` | `0.45` | `2.84e-02` | `4.91e-02` | `[0, 0.927]` |
| `irregular` | `TVD-Tri` | `4.63e-02` | `0.72` | `1.05e-02` | `1.59e-02` | `[-4.4e-14, 1]` |
| `irregular` | `ENO3-Tri` | `4.20e-02` | `0.75` | `9.88e-03` | `1.43e-02` | `[-3.3e-07, 1.0001]` |
| `irregular` | `WENO-Tri` | `4.15e-02` | `0.76` | `9.81e-03` | `1.41e-02` | `[-0.00042, 1.0006]` |

</details>

### 3. Kurganov–Petrova–Popov (KPP) Rotating Wave (`problem3`)
- **Equation & Domain:** Nonconvex scalar conservation law $u_t + (\sin u)_x + (\cos u)_y = 0$ on $(-2.5, 2.5)^2$, $T = 1.0$, $\mathrm{CFL} = 0.45$, global Rusanov flux ($\alpha = 1$), grids $N \in \{100, 200, 400, 800\}$.
- **Setup:** $u_0 = \frac{7\pi}{2}$ inside $\{x^2+y^2 \le 1\}$ (exact disk–cell area fractions), $u_0 = \frac{\pi}{4}$ outside and on all boundaries. Because no closed-form elementary analytical solution exists in 2D at $T=1$, solutions are benchmarked against a cell-averaged restriction of a $1600 \times 1600$ 1st-order monotone Rusanov reference.
- **Variants:** `1st-Order`, `TVD-minmod`, `ENO3`, `WENO5-z`.
- **What it tests:** Nonconvex composite-wave entropy stability. Because $\sin u$ and $\cos u$ change convexity across $[\pi/4, 7\pi/2]$, low-dissipation dimension-by-dimension high-order reconstructions (`ENO3`, `WENO5-z`) allow spurious corner twisting in the composite spiral wave to persist or grow under refinement, producing larger discrepancy against the monotone entropy reference than more dissipative schemes.

<details>
<summary>Representative L¹ discrepancy vs. 1600² monotone Rusanov reference</summary>

| Scheme | $100^2$ | $200^2$ | $400^2$ | $800^2$ | Bounds at $800^2$ |
|---|---:|---:|---:|---:|---|
| `1st-Order` | `4.05` | `2.25` | `1.13` | `0.431` | `[0.785398, 10.9956]` |
| `TVD-minmod` | `1.43` | `0.667` | `0.246` | `0.280` | `[0.785398, 10.9956]` |
| `ENO3` | `1.76` | `0.977` | `0.554` | `0.534` | `[0.785398, 10.9956]` |
| `WENO5-z` | `2.22` | `1.61` | `1.31` | `1.26` | `[0.785394, 10.9956]` |

</details>

---

## CLI Usage & Outputs

Each executable runs all 4 canonical variants by default and accepts serial (`./build/problemX`) or MPI (`mpiexec -n 4 ./build_mpi/problemX`) execution:

```bash
./build/problem1                                    # Diagonal band across 40..320 grids
./build/problem2 --mesh nested                      # Nested 1-to-4 smoothed Delaunay hierarchy
./build/problem2 --mesh irregular                   # Unsmoothed irregular Delaunay hierarchy
./build/problem3                                    # KPP 100..800 vs 1600^2 monotone reference
mpiexec -n 4 ./build_mpi/problem3 --ref 800         # Faster MPI run using an 800^2 reference
```

### Runtime Flags
- **All drivers:** `--scheme first|tvd|eno|weno|all` (default `all`), `--cfl <val>` (default `0.45`), `--t <t_end>`, `--out <dir>`
- **`problem1`:** `--w <half_width>` (default `0.2`), `--grids <n1,n2,...>` (default `40,80,160,320`)
- **`problem2`:** `--mesh nested|irregular` (default `nested`), `--tri-jitter <0..0.45>`, `--tri-smooth <iters>`, `--refs <r0,r1,...>` (default `0,1,2,3`)
- **`problem3`:** `--grids <n1,n2,...>` (default `100,200,400,800`), `--ref <N>` (`0` disables reference solve; default `1600`)

### Generated Output Files
Results are written under `output/problem{1,2,3}/` (gitignored by default):
- `convergence.csv` (and `convergence_irregular.csv` for Problem 2 `--mesh irregular`): cell counts, mesh sizes, $L^1 / L^2 / L^\infty$ errors and EOCs, conservation drift, and runtimes.
- 1D slice/profile CSVs: normal profiles in `output/problem1/profile_*.csv`; streamwise ($y=0.5$) and cross-stream ($x=2.75$) slices in `output/problem2/slice_*.csv`.
- ParaView time series (`solution.pvd` referencing `.vti` or `.vtu` snapshots) for the finest mesh level of each solver variant. Open any `solution.pvd` directly in ParaView to inspect time evolution.

---

## MPI Parallelism

Distributed execution (`USE_MPI=1` / `-DFVM_USE_MPI=ON`) is implemented in [include/fvm/core/mpi_context.hpp](include/fvm/core/mpi_context.hpp):
- **Structured grids (`CartDecomp2D`):** 2D Cartesian process grid with sequential X-then-Y halo exchanges so diagonal ghost corners required by 2D ENO3/WENO5 stencils are populated automatically across rank boundaries and periodic wraps. Rank 0 gathers full fields for profile extraction and `.vti` export.
- **Unstructured meshes (`UnstructuredDecomp`):** Centroid-sorted owner-computes partition along $x$ with precomputed `MPI_Alltoallv` communication schedules for 2-ring scalar values and 1-ring reconstructed polynomials, avoiding redundant least-squares QR factorizations and $P^2$ reconstructions on remote cells.

---

## Repository Structure

```text
.
├── CMakeLists.txt
├── Makefile
├── apps/
│   ├── problem1_diagonal_band.cpp      # Problem 1 driver (structured diagonal band)
│   ├── problem2_unstructured_tube.cpp  # Problem 2 driver (unstructured tube patch)
│   └── problem3_kpp.cpp                # Problem 3 driver (KPP nonconvex rotating wave)
├── include/fvm/
│   ├── core/                           # Types, ghost-aware fields, convergence tools, MPI context
│   ├── io/                             # ParaView .vti / .vtu / .pvd writers
│   ├── linalg/                         # Dense LU / Householder QR and sparse CSR CG / BiCGSTAB
│   ├── mesh/                           # Exact clipping geometry, structured/unstructured meshes, Bowyer-Watson generator
│   ├── numerics/                       # 1D/2D Gauss quadrature, SSP-RK integrators, upwind & Rusanov fluxes
│   └── solvers/                        # StructuredFVSolver & UnstructuredAdvectionSolver
├── src/                                # Library implementations matching include/fvm/
└── tests/
    ├── test_core.cpp                   # Core infrastructure unit tests
    └── test_solvers.cpp                # Convergence-order, conservation, KPP, and MPI equivalence tests
```
