#include "fvm/mesh/mesh_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace fvm {

namespace {

struct TriIdx {
    Index v[3];
    SignedIndex nb[3];  // neighbor across edge (v[k], v[(k+1)%3]), -1 if outer supertriangle edge
    bool alive = true;
};

/// Returns > 0 if p lies strictly inside the circumcircle of CCW triangle (a, b, c).
Real in_circumcircle(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& p) {
    const Real ax = a.x - p.x;
    const Real ay = a.y - p.y;
    const Real bx = b.x - p.x;
    const Real by = b.y - p.y;
    const Real cx = c.x - p.x;
    const Real cy = c.y - p.y;

    const Real a2 = ax * ax + ay * ay;
    const Real b2 = bx * bx + by * by;
    const Real c2 = cx * cx + cy * cy;

    return a2 * (bx * cy - by * cx)
         - b2 * (ax * cy - ay * cx)
         + c2 * (ax * by - ay * bx);
}

struct CavityEdge {
    Index u;
    Index v;
    SignedIndex ext_tri;
    Index old_tid;
};

std::vector<std::array<Index, 3>> bowyer_watson_delaunay(
    const std::vector<Vec2>& points,
    Real xmin, Real xmax, Real ymin, Real ymax) {
    const Index n_pts = points.size();
    const Real lx = xmax - xmin;
    const Real ly = ymax - ymin;
    const Real span = std::max(lx, ly);
    const Real cx = 0.5 * (xmin + xmax);
    const Real cy = 0.5 * (ymin + ymax);

    // Spatial serpentine ordering so consecutive points are geometrically close -> O(1) walk
    std::vector<Index> order(n_pts);
    for (Index i = 0; i < n_pts; ++i) order[i] = i;
    const Index bins = std::max<Index>(1, static_cast<Index>(std::sqrt(static_cast<Real>(n_pts))));
    std::sort(order.begin(), order.end(), [&](Index ia, Index ib) {
        const auto& pa = points[ia];
        const auto& pb = points[ib];
        const Index row_a = static_cast<Index>(std::clamp((pa.y - ymin) / ly, 0.0, 0.999999) * bins);
        const Index row_b = static_cast<Index>(std::clamp((pb.y - ymin) / ly, 0.0, 0.999999) * bins);
        if (row_a != row_b) return row_a < row_b;
        return (row_a % 2 == 0) ? (pa.x < pb.x) : (pa.x > pb.x);
    });

    // Build augmented vertex list with supertriangle at indices [n_pts, n_pts+1, n_pts+2]
    std::vector<Vec2> pts = points;
    const Index s0 = n_pts;
    const Index s1 = n_pts + 1;
    const Index s2 = n_pts + 2;
    pts.push_back({cx - 20.0 * span, cy - 20.0 * span});
    pts.push_back({cx + 20.0 * span, cy - 20.0 * span});
    pts.push_back({cx, cy + 25.0 * span});

    std::vector<TriIdx> tris;
    tris.reserve(2 * n_pts + 16);
    tris.push_back({{s0, s1, s2}, {-1, -1, -1}, true});  // already CCW

    const Real scale2 = span * span;
    const Real circle_tol = 1.0e-13 * scale2 * scale2;
    const Real walk_tol = -1.0e-13 * scale2;

    Index hint_tri = 0;
    std::vector<Index> bad;
    std::vector<CavityEdge> boundary;
    std::vector<Index> new_ids;
    bad.reserve(32);
    boundary.reserve(32);
    new_ids.reserve(32);

    auto contains_in_vec = [](const std::vector<Index>& vec, Index val) {
        for (Index x : vec) {
            if (x == val) return true;
        }
        return false;
    };

    for (Index pid : order) {
        const Vec2& p = pts[pid];

        // 1. Visibility walk from hint_tri to locate triangle containing p
        if (!tris[hint_tri].alive) {
            for (Index k = tris.size(); k > 0; --k) {
                if (tris[k - 1].alive) {
                    hint_tri = k - 1;
                    break;
                }
            }
        }

        Index cur = hint_tri;
        const Index max_walk = tris.size() + 16;
        for (Index step = 0; step < max_walk; ++step) {
            const auto& t = tris[cur];
            SignedIndex next = -1;
            for (int e = 0; e < 3; ++e) {
                if (orient2d(pts[t.v[e]], pts[t.v[(e + 1) % 3]], p) < walk_tol && t.nb[e] >= 0) {
                    next = t.nb[e];
                    break;
                }
            }
            if (next < 0) break;
            cur = static_cast<Index>(next);
        }

        // Verify circumcircle containment; fallback scan if edge precision caused walk miss
        if (!tris[cur].alive ||
            in_circumcircle(pts[tris[cur].v[0]], pts[tris[cur].v[1]], pts[tris[cur].v[2]], p) <=
                circle_tol) {
            bool found = false;
            for (Index k = tris.size(); k > 0; --k) {
                const Index tid = k - 1;
                if (!tris[tid].alive) continue;
                const auto& t = tris[tid];
                if (in_circumcircle(pts[t.v[0]], pts[t.v[1]], pts[t.v[2]], p) > circle_tol) {
                    cur = tid;
                    found = true;
                    break;
                }
            }
            if (!found) continue;
        }

        // 2. BFS cavity expansion over Delaunay adjacency
        bad.clear();
        bad.push_back(cur);
        for (Index bi = 0; bi < bad.size(); ++bi) {
            const Index tid = bad[bi];
            const auto& t = tris[tid];
            for (int e = 0; e < 3; ++e) {
                const SignedIndex nt_s = t.nb[e];
                if (nt_s < 0) continue;
                const Index nt = static_cast<Index>(nt_s);
                if (!tris[nt].alive || contains_in_vec(bad, nt)) continue;
                const auto& tn = tris[nt];
                if (in_circumcircle(pts[tn.v[0]], pts[tn.v[1]], pts[tn.v[2]], p) > circle_tol) {
                    bad.push_back(nt);
                }
            }
        }

        // 3. Extract cavity boundary edges + external neighbor links
        boundary.clear();
        for (Index tid : bad) {
            const auto& t = tris[tid];
            for (int e = 0; e < 3; ++e) {
                const SignedIndex nt_s = t.nb[e];
                if (nt_s < 0 || !contains_in_vec(bad, static_cast<Index>(nt_s))) {
                    boundary.push_back({t.v[e], t.v[(e + 1) % 3], nt_s, tid});
                }
            }
        }
        for (Index tid : bad) {
            tris[tid].alive = false;
        }

        // 4. Re-triangulate cavity by connecting boundary edges to pid and link neighbors
        new_ids.clear();
        for (const auto& be : boundary) {
            const Index new_tid = tris.size();
            new_ids.push_back(new_tid);
            tris.push_back({{be.u, be.v, pid}, {be.ext_tri, -1, -1}, true});

            if (be.ext_tri >= 0) {
                auto& ext = tris[static_cast<Index>(be.ext_tri)];
                for (int ek = 0; ek < 3; ++ek) {
                    if (ext.nb[ek] == static_cast<SignedIndex>(be.old_tid)) {
                        ext.nb[ek] = static_cast<SignedIndex>(new_tid);
                        break;
                    }
                }
            }
        }

        const Index m = boundary.size();
        for (Index i = 0; i < m; ++i) {
            const Index v_end = boundary[i].v;
            for (Index j = 0; j < m; ++j) {
                if (boundary[j].u == v_end) {
                    tris[new_ids[i]].nb[1] = static_cast<SignedIndex>(new_ids[j]);
                    tris[new_ids[j]].nb[2] = static_cast<SignedIndex>(new_ids[i]);
                    break;
                }
            }
        }

        if (!new_ids.empty()) {
            hint_tri = new_ids.back();
        }
    }

    const Real area_eps = 1.0e-12 * lx * ly / static_cast<Real>(std::max<Index>(1, n_pts));
    std::vector<std::array<Index, 3>> result;
    result.reserve(tris.size());
    for (const auto& t : tris) {
        if (!t.alive) continue;
        if (t.v[0] >= n_pts || t.v[1] >= n_pts || t.v[2] >= n_pts) continue;
        const Real twice_area = orient2d(pts[t.v[0]], pts[t.v[1]], pts[t.v[2]]);
        if (0.5 * std::abs(twice_area) <= area_eps) continue;
        if (twice_area > 0.0) {
            result.push_back({t.v[0], t.v[1], t.v[2]});
        } else {
            result.push_back({t.v[0], t.v[2], t.v[1]});
        }
    }
    return result;
}

}  // namespace

UnstructuredMesh2D generate_structured_tri_mesh(
    Real xmin, Real xmax, Real ymin, Real ymax,
    Index nx, Index ny,
    StructuredTriPattern pattern) {
    if (xmax <= xmin || ymax <= ymin || nx == 0 || ny == 0) {
        throw std::invalid_argument("generate_structured_tri_mesh: invalid dimensions");
    }

    const Real dx = (xmax - xmin) / static_cast<Real>(nx);
    const Real dy = (ymax - ymin) / static_cast<Real>(ny);
    const Real tol = 1.0e-10 * std::min(dx, dy);

    std::vector<Vec2> vertices;
    vertices.reserve((nx + 1) * (ny + 1) + (pattern == StructuredTriPattern::UnionJack ? nx * ny : 0));

    auto vid = [nx](Index i, Index j) -> Index { return j * (nx + 1) + i; };
    for (Index j = 0; j <= ny; ++j) {
        const Real y = ymin + static_cast<Real>(j) * dy;
        for (Index i = 0; i <= nx; ++i) {
            const Real x = xmin + static_cast<Real>(i) * dx;
            vertices.push_back({x, y});
        }
    }

    std::vector<std::vector<Index>> cells;
    cells.reserve(nx * ny * (pattern == StructuredTriPattern::UnionJack ? 4 : 2));

    for (Index j = 0; j < ny; ++j) {
        for (Index i = 0; i < nx; ++i) {
            const Index v00 = vid(i, j);
            const Index v10 = vid(i + 1, j);
            const Index v11 = vid(i + 1, j + 1);
            const Index v01 = vid(i, j + 1);

            if (pattern == StructuredTriPattern::UnionJack) {
                const Index vc = vertices.size();
                vertices.push_back({
                    xmin + (static_cast<Real>(i) + 0.5) * dx,
                    ymin + (static_cast<Real>(j) + 0.5) * dy
                });
                cells.push_back({v00, v10, vc});
                cells.push_back({v10, v11, vc});
                cells.push_back({v11, v01, vc});
                cells.push_back({v01, v00, vc});
            } else {
                const bool use_forward =
                    (pattern == StructuredTriPattern::ForwardSlash) || ((i + j) % 2 == 0);
                if (use_forward) {
                    cells.push_back({v00, v10, v11});
                    cells.push_back({v00, v11, v01});
                } else {
                    cells.push_back({v00, v10, v01});
                    cells.push_back({v10, v11, v01});
                }
            }
        }
    }

    return UnstructuredMesh2D::build(
        std::move(vertices), std::move(cells),
        [&](const Vec2& mid) { return rectangle_boundary_tag(mid, xmin, xmax, ymin, ymax, tol); });
}

UnstructuredMesh2D generate_unstructured_tri_mesh(
    Real xmin, Real xmax, Real ymin, Real ymax,
    Index nx, Index ny,
    Real jitter,
    int smoothing_iters,
    unsigned int seed) {
    if (xmax <= xmin || ymax <= ymin || nx < 2 || ny < 2) {
        throw std::invalid_argument("generate_unstructured_tri_mesh: invalid dimensions (need nx,ny >= 2)");
    }

    const Real dx = (xmax - xmin) / static_cast<Real>(nx);
    const Real dy = (ymax - ymin) / static_cast<Real>(ny);
    const Real tol = 1.0e-10 * std::min(dx, dy);

    std::vector<Vec2> points;
    std::vector<bool> is_boundary;
    points.reserve((nx + 1) * (ny + 1));
    is_boundary.reserve((nx + 1) * (ny + 1));

    // Boundary points (4 edges, avoiding duplicate corners)
    for (Index i = 0; i <= nx; ++i) {
        points.push_back({xmin + static_cast<Real>(i) * dx, ymin});
        is_boundary.push_back(true);
    }
    for (Index j = 1; j <= ny; ++j) {
        points.push_back({xmax, ymin + static_cast<Real>(j) * dy});
        is_boundary.push_back(true);
    }
    for (Index i = nx; i > 0; --i) {
        points.push_back({xmin + static_cast<Real>(i - 1) * dx, ymax});
        is_boundary.push_back(true);
    }
    for (Index j = ny - 1; j >= 1; --j) {
        points.push_back({xmin, ymin + static_cast<Real>(j) * dy});
        is_boundary.push_back(true);
    }

    // Interior points on jittered staggered lattice
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> dist(-0.5, 0.5);
    const Real j_clamped = std::clamp(jitter, 0.0, 0.45);

    for (Index j = 1; j < ny; ++j) {
        const Real x_shift = (j % 2 == 0) ? 0.0 : 0.25 * dx;
        for (Index i = 1; i < nx; ++i) {
            Real x = xmin + static_cast<Real>(i) * dx + x_shift + j_clamped * dx * dist(rng);
            Real y = ymin + static_cast<Real>(j) * dy + j_clamped * dy * dist(rng);
            x = std::clamp(x, xmin + 0.25 * dx, xmax - 0.25 * dx);
            y = std::clamp(y, ymin + 0.25 * dy, ymax - 0.25 * dy);
            points.push_back({x, y});
            is_boundary.push_back(false);
        }
    }

    auto tris = bowyer_watson_delaunay(points, xmin, xmax, ymin, ymax);

    // Area-weighted centroid smoothing (Lloyd-like relaxation) for isotropic triangles
    for (int iter = 0; iter < smoothing_iters; ++iter) {
        std::vector<Vec2> accum(points.size(), {0.0, 0.0});
        std::vector<Real> weight_sum(points.size(), 0.0);

        for (const auto& t : tris) {
            const Vec2& p0 = points[t[0]];
            const Vec2& p1 = points[t[1]];
            const Vec2& p2 = points[t[2]];
            const Real area = 0.5 * std::abs(orient2d(p0, p1, p2));
            const Vec2 tc = (p0 + p1 + p2) / 3.0;
            for (int k = 0; k < 3; ++k) {
                accum[t[k]] += area * tc;
                weight_sum[t[k]] += area;
            }
        }

        for (Index pid = 0; pid < points.size(); ++pid) {
            if (!is_boundary[pid] && weight_sum[pid] > 0.0) {
                points[pid] = accum[pid] / weight_sum[pid];
            }
        }
        tris = bowyer_watson_delaunay(points, xmin, xmax, ymin, ymax);
    }

    std::vector<std::vector<Index>> cells;
    cells.reserve(tris.size());
    for (const auto& t : tris) {
        cells.push_back({t[0], t[1], t[2]});
    }

    return UnstructuredMesh2D::build(
        std::move(points), std::move(cells),
        [&](const Vec2& mid) { return rectangle_boundary_tag(mid, xmin, xmax, ymin, ymax, tol); });
}

UnstructuredMesh2D load_gmsh_v2(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_gmsh_v2: failed to open " + path);
    }

    std::vector<Vec2> vertices;
    std::unordered_map<Index, Index> node_id_to_idx;
    std::vector<std::vector<Index>> cells;
    std::unordered_map<std::uint64_t, int> bnd_edge_tags;

    auto pack_edge = [](Index v0, Index v1) -> std::uint64_t {
        const std::uint64_t a = static_cast<std::uint64_t>(std::min(v0, v1));
        const std::uint64_t b = static_cast<std::uint64_t>(std::max(v0, v1));
        return (a << 32) | b;
    };

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "$Nodes") {
            Index n_nodes = 0;
            in >> n_nodes;
            vertices.reserve(n_nodes);
            for (Index i = 0; i < n_nodes; ++i) {
                Index id = 0;
                Real x = 0.0, y = 0.0, z = 0.0;
                in >> id >> x >> y >> z;
                node_id_to_idx[id] = vertices.size();
                vertices.push_back({x, y});
            }
        } else if (line == "$Elements") {
            Index n_elems = 0;
            in >> n_elems;
            for (Index i = 0; i < n_elems; ++i) {
                Index eid = 0;
                int etype = 0;
                int ntags = 0;
                in >> eid >> etype >> ntags;
                std::vector<int> tags(static_cast<Index>(ntags), 0);
                for (int t = 0; t < ntags; ++t) {
                    in >> tags[static_cast<Index>(t)];
                }
                const int phys_tag = (ntags >= 1 && tags[0] != 0) ? tags[0] : 1;

                if (etype == 1) {  // 2-node line
                    Index n0 = 0, n1 = 0;
                    in >> n0 >> n1;
                    bnd_edge_tags[pack_edge(node_id_to_idx.at(n0), node_id_to_idx.at(n1))] = phys_tag;
                } else if (etype == 2) {  // 3-node triangle
                    Index n0 = 0, n1 = 0, n2 = 0;
                    in >> n0 >> n1 >> n2;
                    cells.push_back({node_id_to_idx.at(n0), node_id_to_idx.at(n1), node_id_to_idx.at(n2)});
                } else if (etype == 3) {  // 4-node quad
                    Index n0 = 0, n1 = 0, n2 = 0, n3 = 0;
                    in >> n0 >> n1 >> n2 >> n3;
                    cells.push_back({
                        node_id_to_idx.at(n0), node_id_to_idx.at(n1),
                        node_id_to_idx.at(n2), node_id_to_idx.at(n3)
                    });
                } else {
                    std::string rest;
                    std::getline(in, rest);
                }
            }
        }
    }

    auto edge_tag_fn = [&bnd_edge_tags, &pack_edge](Index v0, Index v1, const Vec2&) -> int {
        const auto it = bnd_edge_tags.find(pack_edge(v0, v1));
        return (it != bnd_edge_tags.end()) ? it->second : 1;
    };

    return UnstructuredMesh2D::build_with_edge_tags(
        std::move(vertices), std::move(cells), edge_tag_fn);
}

}  // namespace fvm
