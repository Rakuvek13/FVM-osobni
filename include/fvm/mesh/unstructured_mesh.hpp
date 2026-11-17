#pragma once

#include "fvm/core/types.hpp"

#include <functional>
#include <string>
#include <vector>

namespace fvm {

struct UnstructuredCell {
    std::vector<Index> vertex_ids;        // CCW ordered
    std::vector<Index> face_ids;          // edge k connects vertex_ids[k] -> vertex_ids[(k+1)%m]
    std::vector<SignedIndex> neighbor_ids; // -1 if boundary face
    Vec2 centroid;
    Real area = 0.0;
    Real inradius = 0.0;  // 2 * area / perimeter
    Real h_char = 0.0;    // sqrt(area)
    Real diam = 0.0;      // max pairwise distance between vertices of cell K
    Real quality = 0.0;   // shape regularity in (0, 1] (4*sqrt(3)*area / sum(l_e^2) for triangles)
};

struct UnstructuredFace {
    Index v0 = 0;
    Index v1 = 0;
    Index left_cell = 0;
    SignedIndex right_cell = -1;  // -1 if boundary
    Vec2 midpoint;
    Vec2 normal;          // unit outward normal from left_cell (to right_cell or domain exterior)
    Real length = 0.0;
    int boundary_tag = 0; // 0 = interior; nonzero = user/generator boundary ID
};

class UnstructuredMesh2D {
public:
    UnstructuredMesh2D() = default;

    /// Builds mesh topology and geometry from vertices and polygonal cells.
    /// Automatically enforces CCW cell winding.
    /// boundary_tag_fn(face_midpoint) assigns nonzero tags to boundary faces.
    static UnstructuredMesh2D build(
        std::vector<Vec2> vertices,
        std::vector<std::vector<Index>> cell_vertices,
        const std::function<int(const Vec2&)>& boundary_tag_fn = {});

    /// Variant using explicit boundary edge tags: map from sorted (v_min, v_max) if needed
    /// via a callback receiving (v0, v1, midpoint).
    static UnstructuredMesh2D build_with_edge_tags(
        std::vector<Vec2> vertices,
        std::vector<std::vector<Index>> cell_vertices,
        const std::function<int(Index, Index, const Vec2&)>& edge_tag_fn);

    Index num_vertices() const { return vertices_.size(); }
    Index num_cells() const { return cells_.size(); }
    Index num_faces() const { return faces_.size(); }

    const std::vector<Vec2>& vertices() const { return vertices_; }
    const std::vector<UnstructuredCell>& cells() const { return cells_; }
    const std::vector<UnstructuredFace>& faces() const { return faces_; }
    const std::vector<std::vector<Index>>& vertex_to_cells() const { return vertex_to_cells_; }
    const std::vector<Real>& cell_areas() const { return cell_areas_; }

    Real h_min() const { return h_min_; }
    Real h_max() const { return h_max_; }
    Real h_mean() const { return h_mean_; }
    Real h_max_diam() const { return h_max_diam_; }
    Real min_inradius() const { return min_inradius_; }
    Real min_quality() const { return min_quality_; }

    /// Returns the CCW vertex coordinates of cell `cell_id`.
    std::vector<Vec2> cell_polygon(Index cell_id) const;

    /// Uniform 1-to-4 subdivision (triangles via edge midpoints; quads via midpoints+centroid),
    /// preserving boundary tags and triangle quality while halving h_max_diam.
    UnstructuredMesh2D refine_uniform() const;

    /// Distinct cells sharing at least one vertex with `cell_id` (excluding `cell_id`).
    std::vector<Index> cell_vertex_neighbors(Index cell_id) const;

    /// Collects at least `min_neighbors` neighboring cells (excluding `cell_id`) via BFS.
    /// When `use_vertex_neighbors` is true, each BFS layer expands through shared vertices
    /// (giving compact isotropic rings); otherwise expands strictly across faces.
    std::vector<Index> collect_stencil(Index cell_id,
                                       Index min_neighbors,
                                       bool use_vertex_neighbors = true) const;

    /// Integrates f(x, y) over cell `cell_id` using triangle quadrature of degree <= 5.
    Real integrate_over_cell(Index cell_id,
                             const std::function<Real(Real, Real)>& f,
                             int quad_degree = 5) const;

    /// Initializes cell-averaged field values using quadrature over every cell.
    void initialize_cell_averages(std::vector<Real>& field,
                                  const std::function<Real(Real, Real)>& f,
                                  int quad_degree = 5) const;

    /// Verifies areas > 0, face lengths > 0, outward normals, and discrete divergence identity.
    /// Throws std::runtime_error with diagnostic details on failure.
    void verify_geometry(Real tol = 1.0e-11) const;

private:
    std::vector<Vec2> vertices_;
    std::vector<UnstructuredCell> cells_;
    std::vector<UnstructuredFace> faces_;
    std::vector<std::vector<Index>> vertex_to_cells_;
    std::vector<Real> cell_areas_;

    Real h_min_ = 0.0;
    Real h_max_ = 0.0;
    Real h_mean_ = 0.0;
    Real h_max_diam_ = 0.0;
    Real min_inradius_ = 0.0;
    Real min_quality_ = 0.0;
};

}  // namespace fvm
