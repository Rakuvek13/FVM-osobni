#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/structured_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef FVM_USE_MPI
#include <mpi.h>
#endif

namespace fvm::mpi {

/// Splits `n_global` items across `num_ranks` nearly evenly.
inline void split_1d(Index n_global, int num_ranks, int rank_coord,
                     Index& start_out, Index& count_out) {
    const Index p = static_cast<Index>(num_ranks);
    const Index r = static_cast<Index>(rank_coord);
    const Index base = n_global / p;
    const Index rem = n_global % p;
    count_out = base + (r < rem ? 1 : 0);
    start_out = r * base + std::min(r, rem);
}

struct HaloScratch {
    std::vector<Real> send_left;
    std::vector<Real> send_right;
    std::vector<Real> recv_left;
    std::vector<Real> recv_right;
};

#ifdef FVM_USE_MPI

/// RAII wrapper around MPI_Init / MPI_Finalize.
class MpiGuard {
public:
    MpiGuard(int& argc, char**& argv) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(&argc, &argv);
            owns_init_ = true;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~MpiGuard() {
        if (owns_init_) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Finalize();
            }
        }
    }

    MpiGuard(const MpiGuard&) = delete;
    MpiGuard& operator=(const MpiGuard&) = delete;

    int rank() const { return rank_; }
    int size() const { return size_; }

private:
    bool owns_init_ = false;
    int rank_ = 0;
    int size_ = 1;
};

/// 2D Cartesian process grid and balanced subdomain decomposition.
struct CartDecomp2D {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    std::array<int, 2> dims{1, 1};      // {px, py}
    std::array<int, 2> coords{0, 0};    // {rx, ry}
    std::array<bool, 2> periodic{false, false};

    int nbr_left = MPI_PROC_NULL;
    int nbr_right = MPI_PROC_NULL;
    int nbr_bottom = MPI_PROC_NULL;
    int nbr_top = MPI_PROC_NULL;

    Index nx_global = 0, ny_global = 0;
    Index i0 = 0, j0 = 0;               // global starting indices
    Index nx_local = 0, ny_local = 0;

    static CartDecomp2D create(Index nx, Index ny,
                               bool periodic_x, bool periodic_y,
                               MPI_Comm base_comm = MPI_COMM_WORLD) {
        CartDecomp2D d;
        d.nx_global = nx;
        d.ny_global = ny;
        d.periodic = {periodic_x, periodic_y};

        int world_size = 1;
        MPI_Comm_size(base_comm, &world_size);

        int raw_dims[2] = {0, 0};
        MPI_Dims_create(world_size, 2, raw_dims);
        if ((nx > ny && raw_dims[0] < raw_dims[1]) ||
            (ny > nx && raw_dims[0] > raw_dims[1])) {
            std::swap(raw_dims[0], raw_dims[1]);
        }
        d.dims = {raw_dims[0], raw_dims[1]};

        int periods[2] = {periodic_x ? 1 : 0, periodic_y ? 1 : 0};
        MPI_Cart_create(base_comm, 2, raw_dims, periods, 1 /* reorder */, &d.comm);

        MPI_Comm_rank(d.comm, &d.rank);
        MPI_Comm_size(d.comm, &d.size);
        int raw_coords[2] = {0, 0};
        MPI_Cart_coords(d.comm, d.rank, 2, raw_coords);
        d.coords = {raw_coords[0], raw_coords[1]};

        MPI_Cart_shift(d.comm, 0, 1, &d.nbr_left, &d.nbr_right);
        MPI_Cart_shift(d.comm, 1, 1, &d.nbr_bottom, &d.nbr_top);

        split_1d(nx, d.dims[0], d.coords[0], d.i0, d.nx_local);
        split_1d(ny, d.dims[1], d.coords[1], d.j0, d.ny_local);
        return d;
    }

    ~CartDecomp2D() {
        if (comm != MPI_COMM_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Comm_free(&comm);
            }
        }
    }

    CartDecomp2D() = default;
    CartDecomp2D(const CartDecomp2D&) = delete;
    CartDecomp2D& operator=(const CartDecomp2D&) = delete;
    CartDecomp2D(CartDecomp2D&& other) noexcept { *this = std::move(other); }
    CartDecomp2D& operator=(CartDecomp2D&& other) noexcept {
        if (this != &other) {
            if (comm != MPI_COMM_NULL) {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized) MPI_Comm_free(&comm);
            }
            comm = other.comm;
            other.comm = MPI_COMM_NULL;
            rank = other.rank; size = other.size;
            dims = other.dims; coords = other.coords;
            periodic = other.periodic;
            nbr_left = other.nbr_left; nbr_right = other.nbr_right;
            nbr_bottom = other.nbr_bottom; nbr_top = other.nbr_top;
            nx_global = other.nx_global; ny_global = other.ny_global;
            i0 = other.i0; j0 = other.j0;
            nx_local = other.nx_local; ny_local = other.ny_local;
        }
        return *this;
    }

    bool on_physical_left()   const { return !periodic[0] && coords[0] == 0; }
    bool on_physical_right()  const { return !periodic[0] && coords[0] + 1 == dims[0]; }
    bool on_physical_bottom() const { return !periodic[1] && coords[1] == 0; }
    bool on_physical_top()    const { return !periodic[1] && coords[1] + 1 == dims[1]; }

    StructuredMesh2D make_local_mesh(Real xmin_g, Real xmax_g,
                                     Real ymin_g, Real ymax_g,
                                     Index nghost) const {
        const Real dx = (xmax_g - xmin_g) / static_cast<Real>(nx_global);
        const Real dy = (ymax_g - ymin_g) / static_cast<Real>(ny_global);
        const Real x0 = xmin_g + static_cast<Real>(i0) * dx;
        const Real x1 = (i0 + nx_local == nx_global) ? xmax_g : (xmin_g + static_cast<Real>(i0 + nx_local) * dx);
        const Real y0 = ymin_g + static_cast<Real>(j0) * dy;
        const Real y1 = (j0 + ny_local == ny_global) ? ymax_g : (ymin_g + static_cast<Real>(j0 + ny_local) * dy);
        return StructuredMesh2D(x0, x1, y0, y1, nx_local, ny_local, nghost);
    }

    Real allreduce_sum(Real v) const {
        Real out = v;
        if (comm != MPI_COMM_NULL && size > 1) {
            MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, comm);
        }
        return out;
    }

    Real allreduce_min(Real v) const {
        Real out = v;
        if (comm != MPI_COMM_NULL && size > 1) {
            MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MIN, comm);
        }
        return out;
    }

    Real allreduce_max(Real v) const {
        Real out = v;
        if (comm != MPI_COMM_NULL && size > 1) {
            MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
        }
        return out;
    }

    FieldStats allreduce_stats(const FieldStats& local) const {
        if (comm == MPI_COMM_NULL || size == 1) {
            return local;
        }
        FieldStats out;
        out.min_val = allreduce_min(local.min_val);
        out.max_val = allreduce_max(local.max_val);
        out.total_mass = allreduce_sum(local.total_mass);
        return out;
    }
};

/// Sequential X-then-Y halo exchange + physical boundary conditions (automatically populates corners).
inline void exchange_halos_and_apply_bcs(const StructuredMesh2D& mesh,
                                         StructuredField2D& field,
                                         const StructuredBoundaryConfig& bc,
                                         const CartDecomp2D& decomp,
                                         Real t,
                                         HaloScratch* scratch = nullptr) {
    const Index nx = field.nx();
    const Index ny = field.ny();
    const Index g = field.nghost();
    if (g == 0) {
        return;
    }
    if (decomp.size > 1 && (nx < g || ny < g)) {
        throw std::runtime_error(
            "Subdomain smaller than ghost width; use fewer MPI ranks for this grid");
    }

    const SignedIndex nx_s = static_cast<SignedIndex>(nx);
    const SignedIndex ny_s = static_cast<SignedIndex>(ny);
    const SignedIndex g_s = static_cast<SignedIndex>(g);

    // Phase 1: Physical X BCs + interior/periodic X halo exchange (rows j in [0, ny))
    mesh.apply_boundary_conditions_x(
        field, bc, t, decomp.on_physical_left(), decomp.on_physical_right());

    const bool need_x_comm =
        (decomp.nbr_left != MPI_PROC_NULL) || (decomp.nbr_right != MPI_PROC_NULL);
    if (need_x_comm) {
        HaloScratch local_scratch;
        HaloScratch& buf = scratch ? *scratch : local_scratch;
        const Index x_count = ny * g;
        buf.send_left.resize(x_count);
        buf.send_right.resize(x_count);
        buf.recv_left.resize(x_count);
        buf.recv_right.resize(x_count);

        for (Index j = 0; j < ny; ++j) {
            const SignedIndex js = static_cast<SignedIndex>(j);
            for (Index k = 0; k < g; ++k) {
                buf.send_left[j * g + k] = field(static_cast<SignedIndex>(k), js);
                buf.send_right[j * g + k] = field(nx_s - g_s + static_cast<SignedIndex>(k), js);
            }
        }

        constexpr int kTagRightward = 10;
        constexpr int kTagLeftward = 11;
        MPI_Request reqs[4];
        MPI_Irecv(buf.recv_left.data(), static_cast<int>(x_count), MPI_DOUBLE,
                  decomp.nbr_left, kTagRightward, decomp.comm, &reqs[0]);
        MPI_Irecv(buf.recv_right.data(), static_cast<int>(x_count), MPI_DOUBLE,
                  decomp.nbr_right, kTagLeftward, decomp.comm, &reqs[1]);
        MPI_Isend(buf.send_right.data(), static_cast<int>(x_count), MPI_DOUBLE,
                  decomp.nbr_right, kTagRightward, decomp.comm, &reqs[2]);
        MPI_Isend(buf.send_left.data(), static_cast<int>(x_count), MPI_DOUBLE,
                  decomp.nbr_left, kTagLeftward, decomp.comm, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (decomp.nbr_left != MPI_PROC_NULL) {
            for (Index j = 0; j < ny; ++j) {
                const SignedIndex js = static_cast<SignedIndex>(j);
                for (Index k = 0; k < g; ++k) {
                    field(-g_s + static_cast<SignedIndex>(k), js) = buf.recv_left[j * g + k];
                }
            }
        }
        if (decomp.nbr_right != MPI_PROC_NULL) {
            for (Index j = 0; j < ny; ++j) {
                const SignedIndex js = static_cast<SignedIndex>(j);
                for (Index k = 0; k < g; ++k) {
                    field(nx_s + static_cast<SignedIndex>(k), js) = buf.recv_right[j * g + k];
                }
            }
        }
    }

    // Phase 2: Physical Y BCs + contiguous full-stride Y halo exchange (covers corners automatically)
    mesh.apply_boundary_conditions_y(
        field, bc, t, decomp.on_physical_bottom(), decomp.on_physical_top());

    const bool need_y_comm =
        (decomp.nbr_bottom != MPI_PROC_NULL) || (decomp.nbr_top != MPI_PROC_NULL);
    if (need_y_comm) {
        const Index stride = field.stride();
        const int y_count = static_cast<int>(g * stride);
        constexpr int kTagUpward = 20;
        constexpr int kTagDownward = 21;

        MPI_Request reqs[4];
        MPI_Irecv(&field(-g_s, -g_s), y_count, MPI_DOUBLE,
                  decomp.nbr_bottom, kTagUpward, decomp.comm, &reqs[0]);
        MPI_Irecv(&field(-g_s, ny_s), y_count, MPI_DOUBLE,
                  decomp.nbr_top, kTagDownward, decomp.comm, &reqs[1]);
        MPI_Isend(&field(-g_s, ny_s - g_s), y_count, MPI_DOUBLE,
                  decomp.nbr_top, kTagUpward, decomp.comm, &reqs[2]);
        MPI_Isend(&field(-g_s, 0), y_count, MPI_DOUBLE,
                  decomp.nbr_bottom, kTagDownward, decomp.comm, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    }
}

/// Gathers interior cells from all Cartesian ranks into a (nx_global x ny_global) field on rank 0.
inline StructuredField2D gather_structured_field(const StructuredField2D& local,
                                                 const CartDecomp2D& decomp,
                                                 Index nghost_out = 0) {
    if (decomp.comm == MPI_COMM_NULL || decomp.size == 1) {
        StructuredField2D out(local.nx(), local.ny(), nghost_out, 0.0);
        for (Index j = 0; j < local.ny(); ++j) {
            for (Index i = 0; i < local.nx(); ++i) {
                const SignedIndex is = static_cast<SignedIndex>(i);
                const SignedIndex js = static_cast<SignedIndex>(j);
                out(is, js) = local(is, js);
            }
        }
        return out;
    }

    const Index nx_loc = local.nx();
    const Index ny_loc = local.ny();
    std::vector<Real> send_buf(nx_loc * ny_loc);
    for (Index j = 0; j < ny_loc; ++j) {
        for (Index i = 0; i < nx_loc; ++i) {
            send_buf[j * nx_loc + i] = local(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j));
        }
    }

    std::vector<int> counts, displs;
    std::vector<Real> recv_buf;
    if (decomp.rank == 0) {
        counts.resize(static_cast<Index>(decomp.size));
        displs.resize(static_cast<Index>(decomp.size));
        int offset = 0;
        for (int r = 0; r < decomp.size; ++r) {
            int rc[2] = {0, 0};
            MPI_Cart_coords(decomp.comm, r, 2, rc);
            Index rx0 = 0, rx_n = 0, ry0 = 0, ry_n = 0;
            split_1d(decomp.nx_global, decomp.dims[0], rc[0], rx0, rx_n);
            split_1d(decomp.ny_global, decomp.dims[1], rc[1], ry0, ry_n);
            counts[static_cast<Index>(r)] = static_cast<int>(rx_n * ry_n);
            displs[static_cast<Index>(r)] = offset;
            offset += counts[static_cast<Index>(r)];
        }
        recv_buf.resize(static_cast<Index>(offset));
    }

    MPI_Gatherv(send_buf.data(), static_cast<int>(send_buf.size()), MPI_DOUBLE,
                recv_buf.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, decomp.comm);

    StructuredField2D global_field;
    if (decomp.rank == 0) {
        global_field = StructuredField2D(decomp.nx_global, decomp.ny_global, nghost_out, 0.0);
        for (int r = 0; r < decomp.size; ++r) {
            int rc[2] = {0, 0};
            MPI_Cart_coords(decomp.comm, r, 2, rc);
            Index rx0 = 0, rx_n = 0, ry0 = 0, ry_n = 0;
            split_1d(decomp.nx_global, decomp.dims[0], rc[0], rx0, rx_n);
            split_1d(decomp.ny_global, decomp.dims[1], rc[1], ry0, ry_n);
            const Real* src = &recv_buf[static_cast<Index>(displs[static_cast<Index>(r)])];
            for (Index j = 0; j < ry_n; ++j) {
                for (Index i = 0; i < rx_n; ++i) {
                    global_field(static_cast<SignedIndex>(rx0 + i),
                                 static_cast<SignedIndex>(ry0 + j)) = src[j * rx_n + i];
                }
            }
        }
    }
    return global_field;
}

/// Scatters interior values from global field on rank 0 into each rank's local field.
inline void scatter_structured_field(const StructuredField2D& global_on_root,
                                     StructuredField2D& local_out,
                                     const CartDecomp2D& decomp) {
    if (decomp.comm == MPI_COMM_NULL || decomp.size == 1) {
        for (Index j = 0; j < local_out.ny(); ++j) {
            for (Index i = 0; i < local_out.nx(); ++i) {
                const SignedIndex is = static_cast<SignedIndex>(i);
                const SignedIndex js = static_cast<SignedIndex>(j);
                local_out(is, js) = global_on_root(is, js);
            }
        }
        return;
    }

    std::vector<int> counts, displs;
    std::vector<Real> send_buf;
    if (decomp.rank == 0) {
        counts.resize(static_cast<Index>(decomp.size));
        displs.resize(static_cast<Index>(decomp.size));
        int offset = 0;
        for (int r = 0; r < decomp.size; ++r) {
            int rc[2] = {0, 0};
            MPI_Cart_coords(decomp.comm, r, 2, rc);
            Index rx0 = 0, rx_n = 0, ry0 = 0, ry_n = 0;
            split_1d(decomp.nx_global, decomp.dims[0], rc[0], rx0, rx_n);
            split_1d(decomp.ny_global, decomp.dims[1], rc[1], ry0, ry_n);
            counts[static_cast<Index>(r)] = static_cast<int>(rx_n * ry_n);
            displs[static_cast<Index>(r)] = offset;
            offset += counts[static_cast<Index>(r)];
        }
        send_buf.resize(static_cast<Index>(offset));
        for (int r = 0; r < decomp.size; ++r) {
            int rc[2] = {0, 0};
            MPI_Cart_coords(decomp.comm, r, 2, rc);
            Index rx0 = 0, rx_n = 0, ry0 = 0, ry_n = 0;
            split_1d(decomp.nx_global, decomp.dims[0], rc[0], rx0, rx_n);
            split_1d(decomp.ny_global, decomp.dims[1], rc[1], ry0, ry_n);
            Real* dst = &send_buf[static_cast<Index>(displs[static_cast<Index>(r)])];
            for (Index j = 0; j < ry_n; ++j) {
                for (Index i = 0; i < rx_n; ++i) {
                    dst[j * rx_n + i] = global_on_root(static_cast<SignedIndex>(rx0 + i),
                                                       static_cast<SignedIndex>(ry0 + j));
                }
            }
        }
    }

    std::vector<Real> recv_buf(local_out.nx() * local_out.ny());
    MPI_Scatterv(send_buf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 recv_buf.data(), static_cast<int>(recv_buf.size()), MPI_DOUBLE, 0, decomp.comm);

    for (Index j = 0; j < local_out.ny(); ++j) {
        for (Index i = 0; i < local_out.nx(); ++i) {
            local_out(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) =
                recv_buf[j * local_out.nx() + i];
        }
    }
}

/// Distributed global L1, L2, Linf error norms across Cartesian subdomains.
inline ErrorNorms compute_error_norms(const StructuredField2D& local_num,
                                      const StructuredField2D& local_ref,
                                      Real cell_area,
                                      const CartDecomp2D& decomp) {
    if (local_num.nx() != local_ref.nx() || local_num.ny() != local_ref.ny()) {
        throw std::invalid_argument("Field dimension mismatch in mpi::compute_error_norms");
    }
    Real local_l1 = 0.0;
    Real local_l2 = 0.0;
    Real local_inf = 0.0;
    for (Index j = 0; j < local_num.ny(); ++j) {
        for (Index i = 0; i < local_num.nx(); ++i) {
            const Real diff = std::abs(
                local_num(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)) -
                local_ref(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j)));
            local_l1 += diff;
            local_l2 += diff * diff;
            local_inf = std::max(local_inf, diff);
        }
    }
    Real sums_in[2] = {local_l1, local_l2};
    Real sums_out[2] = {local_l1, local_l2};
    if (decomp.comm != MPI_COMM_NULL && decomp.size > 1) {
        MPI_Allreduce(sums_in, sums_out, 2, MPI_DOUBLE, MPI_SUM, decomp.comm);
    }
    ErrorNorms e;
    e.l1 = sums_out[0] * cell_area;
    e.l2 = std::sqrt(sums_out[1] * cell_area);
    e.linf = decomp.allreduce_max(local_inf);
    return e;
}

#else  // Serial fallbacks

class MpiGuard {
public:
    MpiGuard(int&, char**&) {}
    int rank() const { return 0; }
    int size() const { return 1; }
};

struct CartDecomp2D {
    int rank = 0;
    int size = 1;
    std::array<int, 2> dims{1, 1};
    std::array<int, 2> coords{0, 0};
    std::array<bool, 2> periodic{false, false};
    Index nx_global = 0, ny_global = 0;
    Index i0 = 0, j0 = 0;
    Index nx_local = 0, ny_local = 0;

    static CartDecomp2D create(Index nx, Index ny, bool periodic_x, bool periodic_y) {
        CartDecomp2D d;
        d.periodic = {periodic_x, periodic_y};
        d.nx_global = nx; d.ny_global = ny;
        d.nx_local = nx; d.ny_local = ny;
        return d;
    }

    bool on_physical_left()   const { return !periodic[0]; }
    bool on_physical_right()  const { return !periodic[0]; }
    bool on_physical_bottom() const { return !periodic[1]; }
    bool on_physical_top()    const { return !periodic[1]; }

    StructuredMesh2D make_local_mesh(Real xmin_g, Real xmax_g,
                                     Real ymin_g, Real ymax_g,
                                     Index nghost) const {
        return StructuredMesh2D(xmin_g, xmax_g, ymin_g, ymax_g, nx_local, ny_local, nghost);
    }

    Real allreduce_sum(Real v) const { return v; }
    Real allreduce_min(Real v) const { return v; }
    Real allreduce_max(Real v) const { return v; }
    FieldStats allreduce_stats(const FieldStats& local) const { return local; }
};

inline void exchange_halos_and_apply_bcs(const StructuredMesh2D& mesh,
                                         StructuredField2D& field,
                                         const StructuredBoundaryConfig& bc,
                                         const CartDecomp2D&,
                                         Real t,
                                         HaloScratch* = nullptr) {
    mesh.apply_boundary_conditions(field, bc, t);
}

inline StructuredField2D gather_structured_field(const StructuredField2D& local,
                                                 const CartDecomp2D&,
                                                 Index nghost_out = 0) {
    StructuredField2D out(local.nx(), local.ny(), nghost_out, 0.0);
    for (Index j = 0; j < local.ny(); ++j) {
        for (Index i = 0; i < local.nx(); ++i) {
            const SignedIndex is = static_cast<SignedIndex>(i);
            const SignedIndex js = static_cast<SignedIndex>(j);
            out(is, js) = local(is, js);
        }
    }
    return out;
}

inline void scatter_structured_field(const StructuredField2D& global_on_root,
                                     StructuredField2D& local_out,
                                     const CartDecomp2D&) {
    for (Index j = 0; j < local_out.ny(); ++j) {
        for (Index i = 0; i < local_out.nx(); ++i) {
            const SignedIndex is = static_cast<SignedIndex>(i);
            const SignedIndex js = static_cast<SignedIndex>(j);
            local_out(is, js) = global_on_root(is, js);
        }
    }
}

inline ErrorNorms compute_error_norms(const StructuredField2D& local_num,
                                      const StructuredField2D& local_ref,
                                      Real cell_area,
                                      const CartDecomp2D&) {
    return fvm::compute_error_norms(local_num, local_ref, cell_area);
}

#endif

}  // namespace fvm::mpi
