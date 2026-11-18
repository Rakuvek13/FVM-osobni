# 2D Structured Finite Volume Method (FVM) Solver Suite

A self-contained C++20 library and benchmark suite for **2D scalar conservation laws** on structured Cartesian grids. With zero external PDE dependencies in the serial build, it compares **six finite-volume reconstruction schemes**—**1st-Order monotone**, **2nd-Order TVD (MUSCL)**, **2nd-Order ENO (`ENO2`)**, **3rd-Order ENO (`ENO3`)**, **3rd-Order WENO (`WENO3`)**, and **5th-Order WENO (`WENO5`)**—with optional OpenMPI domain decomposition and native ParaView XML (`.vti` / `.pvd`) output.

---

## Features

- **Six Reconstruction Schemes with SSP / Explicit RK Time Integration (`StructuredFVSolver`):**
  - `1st-Order`: piecewise-constant upwind / Rusanov with Forward Euler (`nghost = 1`)
  - `TVD`: piecewise-linear MUSCL (`minmod`, `mc`, `vanleer`, `superbee`) with `SSP-RK2` (`nghost = 2`)
  - `ENO2`: 2nd-order adaptive slope reconstruction with 1-point face flux and `SSP-RK2` (`nghost = 2`)
  - `ENO3`: 3rd-order Newton divided-difference reconstruction with 2-point Gauss face quadrature and `SSP-RK3` (`nghost = 3`)
  - `WENO3`: 3rd-order `WENO-Z` / `WENO-JS` reconstruction (`r = 2`, critical-point scale $\varepsilon = \Delta x^2$ for `WENO-Z`) with 2-point Gauss face quadrature and `SSP-RK3` (`nghost = 2`)
  - `WENO5`: 5th-order `WENO-Z` / `WENO-JS` reconstruction (`r = 3`, $\varepsilon = \Delta x^4$ for `WENO-Z`) with 3-point Gauss face quadrature and Shi–Hu–Shu negative-weight splitting (`SSP-RK3` or `RK4`, `nghost = 3`)
- **Self-Contained Numerics & Geometry:**
  - Exact geometric cell-average initialization via polygon half-plane clipping and disk–rectangle overlap (`include/fvm/mesh/geometry_exact.hpp`)
  - 1D Gauss–Legendre quadrature ($n \in \{1,\dots,5\}$) and tensor-product smooth cell-average initialization (`include/fvm/numerics/quadrature.hpp`)
  - Butcher-weight-aware discrete conservation residual $R_M^n$ monitored per time step (`include/fvm/numerics/time_integrator.hpp`)
- **Optional OpenMPI Parallelism:** Compile-time MPI support (`USE_MPI=1` / `-DFVM_USE_MPI=ON`) with 2D Cartesian sequential X-then-Y halo exchanges (populating diagonal corners automatically), matching serial solutions to roundoff precision (`< 1e-14`).
- **ParaView & CSV Output:** Native `.vti` and `.pvd` time series alongside CSV convergence tables and 1D profile slices.

---

## Requirements & Building

### Prerequisites
- C++20 compiler (`clang++` or `g++`)
- GNU `make` or `cmake` ($\ge 3.16$)
- **Optional:** MPI (`mpicxx` and `mpiexec`, e.g. OpenMPI) for distributed execution; [ParaView](https://www.paraview.org/) for visualizing `.pvd`/`.vti` outputs

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

| Scheme | Spatial Reconstruction | Face Quadrature | Required `nghost` | Time Integrator |
|---|---|---|---|---|
| `1st-Order` | Piecewise constant | 1-pt midpoint | 1 | Forward Euler |
| `TVD` | Limited linear slope (`MC` / `Minmod` / `VanLeer` / `Superbee`) | 1-pt midpoint | 2 | `SSP-RK2` |
| `ENO2` | 2-cell adaptive linear slope | 1-pt midpoint | 2 | `SSP-RK2` |
| `ENO3` | 3-cell Newton divided-difference quadratic | 2-pt Gauss–Legendre | 3 | `SSP-RK3` |
| `WENO3` | 3rd-order `WENO-Z` / `WENO-JS` ($r=2$, $\varepsilon_{\text{Z}} = \Delta x^2$) | 2-pt Gauss–Legendre | 2 | `SSP-RK3` |
| `WENO5` | 5th-order `WENO-Z` / `WENO-JS` ($r=3$, Shi–Hu–Shu splitting) | 3-pt Gauss–Legendre | 3 | `SSP-RK3` / `RK4` |

Smooth formal convergence orders (`TVD`, `ENO2`, `ENO3`, `WENO3-Z`, `WENO3-JS`, `WENO5-Z`) and 4-rank MPI pointwise equivalence are verified in [tests/test_solvers.cpp](tests/test_solvers.cpp).

---

## Benchmark Problems

### 1. Diagonal Band Advection (`problem1`)
- **Equation & Domain:** $u_t + u_x + u_y = 0$ on $(0,1)^2$, $T = 1.0$, $\mathrm{CFL} = 0.45$, grids $N \in \{40, 80, 160, 320\}$.
- **Setup:** Stationary $45^\circ$ contact band $u_0(x,y) = \mathbf{1}_{\{|y-x| < 0.2\}}$ initialized via exact cell-area intersection, exact inflow face averages on $x=0$ and $y=0$, and zero-gradient outflow on $x=1, y=1$.
- **Variants:** `1st-Order`, `TVD-mc`, `ENO2`, `ENO3`, `WENO3-z`, `WENO5-z`.
- **What it tests:** Cross-wind numerical diffusion on non-grid-aligned discontinuities. Asymptotically across a linear contact discontinuity, a $p$-th order scheme scales in $L^1$ as $h^{p/(p+1)}$. At $320^2$, `WENO5-z` narrows the 10–90% front width to `0.01187` (vs. `0.09073` for `1st-Order`), while compressive `TVD-mc` (`0.01701`) remains strictly within $[0, 1]$ and outperforms `ENO2`, `ENO3`, and `WENO3-z` on the pure jump (`max_RM <= 1.91e-14` across all runs).

<details>
<summary>4-rank MPI convergence metrics on 40² → 320²</summary>

| Scheme | $L^1$ ($320^2$) | `EOC_1` | $L^2$ ($320^2$) | $d_{10\text{--}90}(x+y=1)$ | Bounds $[u_{\min}, u_{\max}]$ | Time (s) |
|---|---:|---:|---:|---:|---|---:|
| `1st-Order` | `5.78e-02` | `0.47` | `1.28e-01` | `0.09073` | `[8.6e-78, 1]` | `1.264` |
| `TVD-mc` | `8.38e-03` | `0.68` | `4.69e-02` | `0.01701` | `[0, 1]` | `1.847` |
| `ENO2` | `1.80e-02` | `0.60` | `6.62e-02` | `0.03115` | `[2.0e-133, 1]` | `1.734` |
| `ENO3` | `1.06e-02` | `0.67` | `4.91e-02` | `0.01997` | `[1.1e-170, 1]` | `4.706` |
| `WENO3-z` | `1.12e-02` | `0.72` | `5.13e-02` | `0.02074` | `[-2.27e-03, 1.0026]` | `5.217` |
| `WENO5-z` | `5.34e-03` | `0.78` | `3.27e-02` | `0.01187` | `[-1.48e-06, 1]` | `8.967` |

</details>

### 2. Smooth Solid-Body Rotation of a Modulated Wave Packet (`problem2`)
- **Equation & Domain:** $u_t + \partial_x(-2\pi(y - 0.5)u) + \partial_y(2\pi(x - 0.5)u) = 0$ on $(0,1)^2$, $T = 1.0$ (one full $360^\circ$ revolution), $\mathrm{CFL} = 0.45$, grids $N \in \{40, 80, 160, 320\}$.
- **Setup:** Off-center Gaussian-modulated cosine packet $u_0(x,y) = \exp\!\bigl(-((x-0.5)^2 + (y-0.7)^2)/(2\sigma^2)\bigr)\cos(k(x-0.5))$ with $\sigma = 0.06$, $k = 6\pi$, initialized via 5-point Gauss cell averages with `DirichletCellAvg` boundary conditions.
- **Variants:** `1st-Order` (`Euler`), `TVD-mc` (`SSP-RK2`), `ENO2` (`SSP-RK2`), `ENO3` (`SSP-RK3`), `WENO3-z` (`SSP-RK3`), `WENO5-z` (`RK4`).
- **What it tests:** Genuine smooth high-order accuracy, peak-amplitude retention across multiple extrema, and work–precision efficiency. `WENO5-z` at **$80^2$** (`L^1 = 3.15e-04`, `1.04 s`) already beats `TVD-mc` at **$320^2$** (`L^1 = 3.37e-04`, `8.81 s`) in $8.5\times$ less runtime, and `WENO5-z` at **$160^2$** (`L^1 = 1.52e-05`, `6.49 s`) beats both `ENO3` (`6.12e-05`, `18.88 s`) and `WENO3-z` (`1.31e-04`, `19.04 s`) at **$320^2$** (`max_RM <= 4.55e-16` across all runs).

<details>
<summary>4-rank MPI convergence metrics on 40² → 320²</summary>

| Scheme | $L^1$ ($80^2$) | $L^1$ ($160^2$) | $L^1$ ($320^2$) | `EOC_1` ($320^2$) | $L^\infty$ ($320^2$) | $R_{\text{peak}}$ ($320^2$) | Time ($320^2$, s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1st-Order` | `1.66e-02` | `1.30e-02` | `9.35e-03` | `0.48` | `5.47e-01` | `0.4639` | `4.330` |
| `TVD-mc` | `3.30e-03` | `1.13e-03` | `3.37e-04` | `1.75` | `3.25e-02` | `0.9729` | `8.814` |
| `ENO2` | `7.80e-03` | `3.04e-03` | `9.36e-04` | `1.70` | `1.28e-01` | `0.8731` | `7.681` |
| `ENO3` | `2.09e-03` | `3.47e-04` | `6.12e-05` | `2.51` | `4.43e-03` | `0.9972` | `18.876` |
| `WENO3-z` | `4.51e-03` | `9.09e-04` | `1.31e-04` | `2.80` | `4.24e-02` | `0.9584` | `19.035` |
| `WENO5-z` | `3.15e-04` | `1.52e-05` | `2.58e-07` | `5.88` | `1.91e-05` | `0.999989` | `45.362` |

</details>

### 3. Kurganov–Petrova–Popov (KPP) Rotating Wave (`problem3`)
- **Equation & Domain:** Nonconvex scalar conservation law $u_t + (\sin u)_x + (\cos u)_y = 0$ on $(-2.5, 2.5)^2$, $T = 1.0$, $\mathrm{CFL} = 0.45$, global Rusanov flux ($\alpha = 1$), default grids $N \in \{100, 200, 400, 800\}$ against a $1600^2$ 1st-order monotone Rusanov reference.
- **Setup:** $u_0 = \frac{7\pi}{2}$ inside $\{x^2+y^2 \le 1\}$ (exact disk–cell area fractions), $u_0 = \frac{\pi}{4}$ outside and on all boundaries.
- **Variants:** `1st-Order`, `TVD-minmod`, `ENO2`, `ENO3`, `WENO3-z`, `WENO5-z`.
- **What it tests:** Nonconvex composite-wave entropy stability. Because $\sin u$ and $\cos u$ change convexity across $[\pi/4, 7\pi/2]$, low-dissipation dimension-by-dimension high-order reconstructions (`ENO3`, `WENO3-z`, `WENO5-z`) allow spurious composite-wave corner twisting to persist under refinement, producing larger discrepancy against the monotone entropy reference than `TVD-minmod` and `ENO2`.

<details>
<summary>4-rank MPI L¹ discrepancy on 100² → 400² vs. 800² monotone Rusanov reference</summary>

| Scheme | $100^2$ | $200^2$ | $400^2$ | Bounds at $400^2$ | Time ($400^2$, s) |
|---|---:|---:|---:|---|---:|
| `1st-Order` | `3.67` | `1.84` | `0.704` | `[0.785398, 10.9956]` | `0.891` |
| `TVD-minmod` | `1.16` | `0.481` | `0.451` | `[0.785398, 10.9956]` | `1.268` |
| `ENO2` | `1.16` | `0.481` | `0.451` | `[0.785398, 10.9956]` | `1.317` |
| `ENO3` | `1.71` | `1.08` | `0.897` | `[0.785398, 10.9956]` | `2.721` |
| `WENO3-z` | `1.79` | `1.14` | `0.897` | `[0.781279, 10.9998]` | `2.940` |
| `WENO5-z` | `2.27` | `1.82` | `1.68` | `[0.785382, 10.9956]` | `5.549` |

</details>

---

## CLI Usage & Outputs

Each executable runs all 6 canonical schemes by default and accepts serial (`./build/problemX`) or MPI (`mpiexec -n 4 ./build_mpi/problemX`) execution:

```bash
./build/problem1                                    # Diagonal band across 40..320 grids
./build/problem2                                    # Smooth solid-body rotation across 40..320 grids
./build/problem3                                    # KPP 100..800 vs 1600^2 monotone reference
mpiexec -n 4 ./build_mpi/problem3 --grids 100,200,400 --ref 800
```

### Runtime Flags
- **All drivers:** `--scheme first|tvd|eno2|eno3|eno|weno3|weno5|weno|all` (default `all`), `--cfl <val>` (default `0.45`), `--t <t_end>`, `--out <dir>`
- **`problem1`:** `--w <half_width>` (default `0.2`), `--grids <n1,n2,...>` (default `40,80,160,320`)
- **`problem2`:** `--sigma <val>` (default `0.06`), `--k <wavenumber>` (default `6*pi`), `--grids <n1,n2,...>` (default `40,80,160,320`)
- **`problem3`:** `--grids <n1,n2,...>` (default `100,200,400,800`), `--ref <N>` (`0` disables reference solve; default `1600`)

### Generated Output Files
Results are written under `output/problem{1,2,3}/` (gitignored by default):
- `convergence.csv`: cell counts, mesh sizes, $L^1 / L^2 / L^\infty$ errors and EOCs, conservation drift, and runtimes.
- 1D slice/profile CSVs: normal profiles in `output/problem1/profile_*.csv`; horizontal slices at $y \approx 0.70$ in `output/problem2/profile_*.csv`.
- ParaView time series (`solution.pvd` referencing `.vti` snapshots) for the finest mesh level of each solver variant.

---

## MPI Parallelism

Distributed execution (`USE_MPI=1` / `-DFVM_USE_MPI=ON`) is implemented in [include/fvm/core/mpi_context.hpp](include/fvm/core/mpi_context.hpp):
- **Structured grids (`CartDecomp2D`):** 2D Cartesian process grid with sequential X-then-Y halo exchanges so diagonal ghost corners required by 2D `ENO3`/`WENO3`/`WENO5` stencils are populated automatically across rank boundaries and periodic wraps. Rank 0 gathers full fields for profile extraction and `.vti` export.

---

## Repository Structure

```text
.
├── CMakeLists.txt
├── Makefile
├── apps/
│   ├── problem1_diagonal_band.cpp      # Problem 1 driver (oblique contact discontinuity)
│   ├── problem2_smooth_rotation.cpp    # Problem 2 driver (smooth solid-body wave-packet rotation)
│   └── problem3_kpp.cpp                # Problem 3 driver (KPP nonconvex rotating wave)
├── include/fvm/
│   ├── core/                           # Types, ghost-aware fields, convergence tools, MPI context
│   ├── io/                             # ParaView .vti / .pvd writers
│   ├── mesh/                           # Exact clipping geometry, structured mesh & BCs
│   ├── numerics/                       # 1D Gauss quadrature, RK integrators, upwind & Rusanov fluxes
│   └── solvers/                        # StructuredFVSolver (1st, TVD, ENO2/3, WENO3/5)
├── src/                                # Library implementations matching include/fvm/
└── tests/
    ├── test_core.cpp                   # Core infrastructure unit tests
    └── test_solvers.cpp                # Convergence-order, conservation, KPP, and MPI equivalence tests
```
