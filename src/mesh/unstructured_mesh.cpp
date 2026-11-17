#include "fvm/mesh/unstructured_mesh.hpp"
#include "fvm/numerics/quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fvm {

namespace {

struct EdgeKey {
    Index a;
    Index b;
    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& k) const {
        const std::uint64_t x = static_cast<std::uint64_t>(k.a);
        const std::uint64_t y = static_cast<std::uint64_t>(k.b);
        return std::hash<std::uint64_t>{}((x << 32) ^ (y * 0x9e3779b97f4a7c15ULL));
    }
};

Real polygon_signed_area_and_centroid(const std::vector<Vec2>& vertices,
                                      const std::vector<Index>& poly,
                                      Vec2& centroid_out) {
    const Index m = poly.size();
    if (m < 3) {
        throw std::invalid_argument("Polygon must have at least 3 vertices");
    }
    Real twice_area = 0.0;
    Vec2 c{0.0, 0.0};
    for (Index k = 0; k < m; ++k) {
        const Vec2& p0 = vertices[poly[k]];
        const Vec2& p1 = vertices[poly[(k + 1) % m]];
        const Real cr = cross(p0, p1);
        twice_area += cr;
        c.x += (p0.x + p1.x) * cr;
        c.y += (p0.y + p1.y) * cr;
    }
    const Real area = 0.5 * twice_area;
    if (std::abs(area) > 1.0e-30) {
        centroid_out = c / (6.0 * area);
    } else {
        centroid_out = {0.0, 0.0};
    }
    return area;
}

}  // namespace

UnstructuredMesh2D UnstructuredMesh2D::build(
    std::vector<Vec2> vertices,
    std::vector<std::vector<Index>> cell_vertices,
    const std::function<int(const Vec2&)>& boundary_tag_fn) {
    auto edge_tag_fn = [&boundary_tag_fn](Index, Index, const Vec2& mid) -> int {
        return boundary_tag_fn ? boundary_tag_fn(mid) : 1;
    };
    return build_with_edge_tags(std::move(vertices), std::move(cell_vertices), edge_tag_fn);
}

UnstructuredMesh2D UnstructuredMesh2D::build_with_edge_tags(
    std::vector<Vec2> vertices,
    std::vector<std::vector<Index>> cell_vertices,
    const std::function<int(Index, Index, const Vec2&)>& edge_tag_fn) {
    UnstructuredMesh2D mesh;
    mesh.vertices_ = std::move(vertices);
    const Index n_cells = cell_vertices.size();
    const Index n_verts = mesh.vertices_.size();
    if (n_cells == 0 || n_verts < 3) {
        throw std::invalid_argument("UnstructuredMesh2D: empty mesh");
    }

    mesh.cells_.resize(n_cells);
    mesh.cell_areas_.resize(n_cells);
    mesh.vertex_to_cells_.assign(n_verts, {});

    std::unordered_map<EdgeKey, Index, EdgeKeyHash> edge_map;
    edge_map.reserve(n_cells * 2);

    for (Index c = 0; c < n_cells; ++c) {
        auto& poly = cell_vertices[c];
        for (Index vid : poly) {
            if (vid >= n_verts) {
                throw std::out_of_range("UnstructuredMesh2D: vertex index out of range");
            }
        }
        Vec2 centroid;
        Real signed_area = polygon_signed_area_and_centroid(mesh.vertices_, poly, centroid);
        if (signed_area < 0.0) {
            std::reverse(poly.begin(), poly.end());
            signed_area = -signed_area;
        }
        if (signed_area <= 1.0e-20) {
            throw std::runtime_error("UnstructuredMesh2D: degenerate zero-area cell");
        }

        UnstructuredCell cell;
        cell.vertex_ids = std::move(poly);
        cell.centroid = centroid;
        cell.area = signed_area;
        cell.h_char = std::sqrt(signed_area);

        const Index m = cell.vertex_ids.size();
        cell.face_ids.resize(m);
        cell.neighbor_ids.assign(m, -1);

        Real perimeter = 0.0;
        Real len_sq_sum = 0.0;
        for (Index k = 0; k < m; ++k) {
            const Index v0 = cell.vertex_ids[k];
            const Index v1 = cell.vertex_ids[(k + 1) % m];
            mesh.vertex_to_cells_[v0].push_back(c);

            const Vec2 edge_vec = mesh.vertices_[v1] - mesh.vertices_[v0];
            const Real len = edge_vec.norm();
            if (len <= 1.0e-18) {
                throw std::runtime_error("UnstructuredMesh2D: zero-length edge");
            }
            perimeter += len;
            len_sq_sum += len * len;

            const EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
            auto it = edge_map.find(key);
            if (it == edge_map.end()) {
                UnstructuredFace face;
                face.v0 = v0;
                face.v1 = v1;
                face.left_cell = c;
                face.right_cell = -1;
                face.midpoint = 0.5 * (mesh.vertices_[v0] + mesh.vertices_[v1]);
                face.length = len;
                // Since cell vertices are CCW, edge_vec.perp_cw() points outward from left_cell
                face.normal = edge_vec.perp_cw() / len;
                face.boundary_tag = edge_tag_fn ? edge_tag_fn(v0, v1, face.midpoint) : 1;

                const Index fid = mesh.faces_.size();
                mesh.faces_.push_back(face);
                edge_map.emplace(key, fid);
                cell.face_ids[k] = fid;
            } else {
                const Index fid = it->second;
                UnstructuredFace& face = mesh.faces_[fid];
                if (face.right_cell != -1) {
                    throw std::runtime_error(
                        "UnstructuredMesh2D: non-manifold edge shared by >2 cells");
                }
                face.right_cell = static_cast<SignedIndex>(c);
                face.boundary_tag = 0;  // interior face
                cell.face_ids[k] = fid;
                cell.neighbor_ids[k] = static_cast<SignedIndex>(face.left_cell);
                // Link left_cell's neighbor_ids entry for this face
                UnstructuredCell& left = mesh.cells_[face.left_cell];
                for (Index lk = 0; lk < left.face_ids.size(); ++lk) {
                    if (left.face_ids[lk] == fid) {
                        left.neighbor_ids[lk] = static_cast<SignedIndex>(c);
                        break;
                    }
                }
            }
        }

        cell.inradius = 2.0 * cell.area / perimeter;
        Real max_diam = 0.0;
        for (Index a = 0; a < m; ++a) {
            for (Index b = a + 1; b < m; ++b) {
                const Real d = (mesh.vertices_[cell.vertex_ids[a]] -
                                mesh.vertices_[cell.vertex_ids[b]]).norm();
                max_diam = std::max(max_diam, d);
            }
        }
        cell.diam = max_diam;
        if (m == 3) {
            cell.quality = 4.0 * std::sqrt(3.0) * cell.area / len_sq_sum;
        } else {
            // Isoperimetric quotient 4*pi*A / P^2 normalized so a square (pi/4) scales or raw [0,1]
            cell.quality = 4.0 * PI * cell.area / (perimeter * perimeter);
        }
        mesh.cell_areas_[c] = cell.area;
        mesh.cells_[c] = std::move(cell);
    }

    // Deduplicate vertex_to_cells lists
    for (auto& vcells : mesh.vertex_to_cells_) {
        std::sort(vcells.begin(), vcells.end());
        vcells.erase(std::unique(vcells.begin(), vcells.end()), vcells.end());
    }

    // Compute mesh size summary
    mesh.h_min_ = std::numeric_limits<Real>::infinity();
    mesh.h_max_ = 0.0;
    mesh.h_max_diam_ = 0.0;
    mesh.min_inradius_ = std::numeric_limits<Real>::infinity();
    mesh.min_quality_ = std::numeric_limits<Real>::infinity();
    Real h_sum = 0.0;
    for (const auto& c : mesh.cells_) {
        mesh.h_min_ = std::min(mesh.h_min_, c.h_char);
        mesh.h_max_ = std::max(mesh.h_max_, c.h_char);
        mesh.h_max_diam_ = std::max(mesh.h_max_diam_, c.diam);
        mesh.min_inradius_ = std::min(mesh.min_inradius_, c.inradius);
        mesh.min_quality_ = std::min(mesh.min_quality_, c.quality);
        h_sum += c.h_char;
    }
    mesh.h_mean_ = h_sum / static_cast<Real>(n_cells);

    return mesh;
}

std::vector<Vec2> UnstructuredMesh2D::cell_polygon(Index cell_id) const {
    const auto& cell = cells_.at(cell_id);
    std::vector<Vec2> poly;
    poly.reserve(cell.vertex_ids.size());
    for (Index vid : cell.vertex_ids) {
        poly.push_back(vertices_[vid]);
    }
    return poly;
}

UnstructuredMesh2D UnstructuredMesh2D::refine_uniform() const {
    std::vector<Vec2> new_verts = vertices_;
    std::unordered_map<EdgeKey, Index, EdgeKeyHash> edge_mid_map;
    edge_mid_map.reserve(faces_.size());

    std::unordered_map<EdgeKey, int, EdgeKeyHash> child_bnd_tags;
    child_bnd_tags.reserve(faces_.size());

    auto get_midpoint = [&](Index a, Index b) -> Index {
        const EdgeKey key{std::min(a, b), std::max(a, b)};
        auto it = edge_mid_map.find(key);
        if (it != edge_mid_map.end()) {
            return it->second;
        }
        const Index mid_id = new_verts.size();
        new_verts.push_back(0.5 * (vertices_[a] + vertices_[b]));
        edge_mid_map.emplace(key, mid_id);
        return mid_id;
    };

    for (const auto& f : faces_) {
        const Index mid_id = get_midpoint(f.v0, f.v1);
        if (f.right_cell < 0) {
            child_bnd_tags[{std::min(f.v0, mid_id), std::max(f.v0, mid_id)}] = f.boundary_tag;
            child_bnd_tags[{std::min(f.v1, mid_id), std::max(f.v1, mid_id)}] = f.boundary_tag;
        }
    }

    std::vector<std::vector<Index>> new_cells;
    new_cells.reserve(cells_.size() * 4);

    for (const auto& c : cells_) {
        if (c.vertex_ids.size() == 3) {
            const Index v0 = c.vertex_ids[0];
            const Index v1 = c.vertex_ids[1];
            const Index v2 = c.vertex_ids[2];
            const Index m01 = get_midpoint(v0, v1);
            const Index m12 = get_midpoint(v1, v2);
            const Index m20 = get_midpoint(v2, v0);
            new_cells.push_back({v0, m01, m20});
            new_cells.push_back({m01, v1, m12});
            new_cells.push_back({m20, m12, v2});
            new_cells.push_back({m01, m12, m20});
        } else if (c.vertex_ids.size() == 4) {
            const Index v0 = c.vertex_ids[0];
            const Index v1 = c.vertex_ids[1];
            const Index v2 = c.vertex_ids[2];
            const Index v3 = c.vertex_ids[3];
            const Index m01 = get_midpoint(v0, v1);
            const Index m12 = get_midpoint(v1, v2);
            const Index m23 = get_midpoint(v2, v3);
            const Index m30 = get_midpoint(v3, v0);
            const Index mc = new_verts.size();
            new_verts.push_back(c.centroid);
            new_cells.push_back({v0, m01, mc, m30});
            new_cells.push_back({m01, v1, m12, mc});
            new_cells.push_back({mc, m12, v2, m23});
            new_cells.push_back({m30, mc, m23, v3});
        } else {
            throw std::runtime_error("refine_uniform only supports triangular and quad meshes");
        }
    }

    auto tag_fn = [&child_bnd_tags](Index v0, Index v1, const Vec2&) -> int {
        const auto it = child_bnd_tags.find({std::min(v0, v1), std::max(v0, v1)});
        return (it != child_bnd_tags.end()) ? it->second : 1;
    };

    return UnstructuredMesh2D::build_with_edge_tags(
        std::move(new_verts), std::move(new_cells), tag_fn);
}

std::vector<Index> UnstructuredMesh2D::cell_vertex_neighbors(Index cell_id) const {
    const auto& cell = cells_.at(cell_id);
    std::vector<Index> neighbors;
    neighbors.reserve(16);
    for (Index vid : cell.vertex_ids) {
        for (Index nc : vertex_to_cells_[vid]) {
            if (nc != cell_id) {
                neighbors.push_back(nc);
            }
        }
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    return neighbors;
}

std::vector<Index> UnstructuredMesh2D::collect_stencil(Index cell_id,
                                                       Index min_neighbors,
                                                       bool use_vertex_neighbors) const {
    if (cell_id >= cells_.size()) {
        throw std::out_of_range("collect_stencil: cell_id out of range");
    }
    std::vector<Index> stencil;
    stencil.reserve(min_neighbors + 6);
    std::unordered_set<Index> visited;
    visited.insert(cell_id);

    std::queue<Index> q;
    q.push(cell_id);

    while (!q.empty() && stencil.size() < min_neighbors) {
        const Index layer_size = q.size();
        for (Index li = 0; li < layer_size; ++li) {
            const Index cur = q.front();
            q.pop();

            // Always enqueue direct face neighbors first for better geometric conditioning
            for (SignedIndex nb : cells_[cur].neighbor_ids) {
                if (nb >= 0) {
                    const Index nbu = static_cast<Index>(nb);
                    if (visited.insert(nbu).second) {
                        stencil.push_back(nbu);
                        q.push(nbu);
                    }
                }
            }
            if (use_vertex_neighbors) {
                for (Index vid : cells_[cur].vertex_ids) {
                    for (Index nbu : vertex_to_cells_[vid]) {
                        if (visited.insert(nbu).second) {
                            stencil.push_back(nbu);
                            q.push(nbu);
                        }
                    }
                }
            }
        }
    }
    return stencil;
}

Real UnstructuredMesh2D::integrate_over_cell(Index cell_id,
                                             const std::function<Real(Real, Real)>& f,
                                             int quad_degree) const {
    const auto& cell = cells_.at(cell_id);
    const auto qpts = quadrature::triangle_rule(quad_degree);

    if (cell.vertex_ids.size() == 3) {
        const Vec2& p0 = vertices_[cell.vertex_ids[0]];
        const Vec2& p1 = vertices_[cell.vertex_ids[1]];
        const Vec2& p2 = vertices_[cell.vertex_ids[2]];
        Real integral = 0.0;
        for (const auto& qp : qpts) {
            const Vec2 x = qp.lambda1 * p0 + qp.lambda2 * p1 + qp.lambda3 * p2;
            integral += qp.weight * f(x.x, x.y);
        }
        return integral * cell.area;
    }

    // General polygon: sub-triangulate around cell centroid
    Real total = 0.0;
    const Vec2& pc = cell.centroid;
    const Index m = cell.vertex_ids.size();
    for (Index k = 0; k < m; ++k) {
        const Vec2& p0 = vertices_[cell.vertex_ids[k]];
        const Vec2& p1 = vertices_[cell.vertex_ids[(k + 1) % m]];
        const Real sub_area = 0.5 * orient2d(pc, p0, p1);
        Real sub_val = 0.0;
        for (const auto& qp : qpts) {
            const Vec2 x = qp.lambda1 * pc + qp.lambda2 * p0 + qp.lambda3 * p1;
            sub_val += qp.weight * f(x.x, x.y);
        }
        total += sub_area * sub_val;
    }
    return total;
}

void UnstructuredMesh2D::initialize_cell_averages(std::vector<Real>& field,
                                                  const std::function<Real(Real, Real)>& f,
                                                  int quad_degree) const {
    field.resize(cells_.size());
    for (Index c = 0; c < cells_.size(); ++c) {
        field[c] = integrate_over_cell(c, f, quad_degree) / cells_[c].area;
    }
}

void UnstructuredMesh2D::verify_geometry(Real tol) const {
    for (Index f = 0; f < faces_.size(); ++f) {
        const auto& face = faces_[f];
        if (face.length <= 0.0) {
            throw std::runtime_error("verify_geometry: non-positive face length");
        }
        if (std::abs(face.normal.norm() - 1.0) > tol) {
            throw std::runtime_error("verify_geometry: non-unit face normal");
        }
        if (face.right_cell >= 0) {
            const Vec2& xL = cells_[face.left_cell].centroid;
            const Vec2& xR = cells_[static_cast<Index>(face.right_cell)].centroid;
            if (dot(xR - xL, face.normal) <= 0.0) {
                std::ostringstream oss;
                oss << "verify_geometry: inward normal on interior face " << f;
                throw std::runtime_error(oss.str());
            }
        } else if (face.boundary_tag == 0) {
            throw std::runtime_error("verify_geometry: boundary face has boundary_tag == 0");
        }
    }

    for (Index c = 0; c < cells_.size(); ++c) {
        const auto& cell = cells_[c];
        if (cell.area <= 0.0 || cell.inradius <= 0.0) {
            throw std::runtime_error("verify_geometry: non-positive cell area/inradius");
        }
        Vec2 div_sum{0.0, 0.0};
        for (Index k = 0; k < cell.face_ids.size(); ++k) {
            const auto& face = faces_[cell.face_ids[k]];
            const Real sign = (face.left_cell == c) ? 1.0 : -1.0;
            div_sum += (sign * face.length) * face.normal;
        }
        if (div_sum.norm() > tol * std::sqrt(cell.area)) {
            std::ostringstream oss;
            oss << "verify_geometry: discrete divergence identity failed on cell " << c
                << " residual=" << div_sum.norm();
            throw std::runtime_error(oss.str());
        }
    }
}

}  // namespace fvm
