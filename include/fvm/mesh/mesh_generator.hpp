#pragma once

#include "fvm/core/types.hpp"
#include "fvm/mesh/unstructured_mesh.hpp"

#include <string>

namespace fvm {

enum class StructuredTriPattern {
    ForwardSlash,
    Alternating,
    UnionJack
};

/// Boundary tags automatically assigned on rectangular domains [xmin, xmax] x [ymin, ymax]:
///   1 = Left (x = xmin), 2 = Right (x = xmax), 3 = Bottom (y = ymin), 4 = Top (y = ymax).
inline int rectangle_boundary_tag(const Vec2& p,
                                  Real xmin, Real xmax,
                                  Real ymin, Real ymax,
                                  Real tol = 1.0e-10) {
    if (std::abs(p.x - xmin) <= tol) return 1;
    if (std::abs(p.x - xmax) <= tol) return 2;
    if (std::abs(p.y - ymin) <= tol) return 3;
    if (std::abs(p.y - ymax) <= tol) return 4;
    return 0;
}

/// Generates a structured rectangular grid split into triangles.
UnstructuredMesh2D generate_structured_tri_mesh(
    Real xmin, Real xmax, Real ymin, Real ymax,
    Index nx, Index ny,
    StructuredTriPattern pattern = StructuredTriPattern::Alternating);

/// Generates an unstructured isotropic Delaunay triangular mesh on [xmin, xmax] x [ymin, ymax]
/// via Bowyer-Watson triangulation + interior Lloyd/centroid smoothing.
UnstructuredMesh2D generate_unstructured_tri_mesh(
    Real xmin, Real xmax, Real ymin, Real ymax,
    Index nx, Index ny,
    Real jitter = 0.35,
    int smoothing_iters = 5,
    unsigned int seed = 42);

/// Reads a 2D ASCII Gmsh .msh (format v2.2) mesh file (triangles and/or quads).
UnstructuredMesh2D load_gmsh_v2(const std::string& path);

}  // namespace fvm
