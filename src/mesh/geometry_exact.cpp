#include "fvm/mesh/geometry_exact.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fvm::geom {

std::vector<Vec2> clip_polygon_halfplane(const std::vector<Vec2>& poly, Real a, Real b, Real c) {
    if (poly.empty()) {
        return {};
    }
    auto eval = [a, b, c](const Vec2& p) { return a * p.x + b * p.y + c; };

    std::vector<Vec2> out;
    out.reserve(poly.size() + 2);

    const Index n = poly.size();
    for (Index i = 0; i < n; ++i) {
        const Vec2& A = poly[i];
        const Vec2& B = poly[(i + 1) % n];
        const Real fA = eval(A);
        const Real fB = eval(B);
        const bool inA = (fA >= 0.0);
        const bool inB = (fB >= 0.0);

        if (inA && inB) {
            out.push_back(B);
        } else if (inA && !inB) {
            const Real t = fA / (fA - fB);
            out.push_back(A + t * (B - A));
        } else if (!inA && inB) {
            const Real t = fA / (fA - fB);
            out.push_back(A + t * (B - A));
            out.push_back(B);
        }
    }
    return out;
}

Real polygon_area(const std::vector<Vec2>& poly) {
    if (poly.size() < 3) {
        return 0.0;
    }
    Real twice_area = 0.0;
    const Index n = poly.size();
    for (Index i = 0; i < n; ++i) {
        twice_area += cross(poly[i], poly[(i + 1) % n]);
    }
    return 0.5 * std::abs(twice_area);
}

Real intersection_area_box(const std::vector<Vec2>& poly,
                           Real xmin, Real xmax, Real ymin, Real ymax) {
    if (xmax <= xmin || ymax <= ymin) {
        return 0.0;
    }
    auto clipped = clip_polygon_halfplane(poly, 1.0, 0.0, -xmin);   // x >= xmin
    clipped = clip_polygon_halfplane(clipped, -1.0, 0.0, xmax);     // x <= xmax
    clipped = clip_polygon_halfplane(clipped, 0.0, 1.0, -ymin);     // y >= ymin
    clipped = clip_polygon_halfplane(clipped, 0.0, -1.0, ymax);     // y <= ymax
    return polygon_area(clipped);
}

Real intersection_area_diagonal_band(const std::vector<Vec2>& poly, Real w) {
    if (w <= 0.0) {
        return 0.0;
    }
    // |y - x| <= w  <=>  y - x + w >= 0  AND  x - y + w >= 0
    auto clipped = clip_polygon_halfplane(poly, -1.0, 1.0, w);
    clipped = clip_polygon_halfplane(clipped, 1.0, -1.0, w);
    return polygon_area(clipped);
}

Real interval_overlap_fraction(Real a, Real b, Real c, Real d) {
    if (b <= a || d <= c) {
        return 0.0;
    }
    const Real lo = std::max(a, c);
    const Real hi = std::min(b, d);
    return (hi > lo) ? (hi - lo) / (b - a) : 0.0;
}

namespace {

/// Exact intersection area in the first quadrant [a, b] x [c, d] with 0 <= a < b, 0 <= c < d
/// and disk x^2 + y^2 <= R^2.
Real disk_quadrant_rect_area(Real a, Real b, Real c, Real d, Real R) {
    if (a >= b || c >= d || R <= 0.0 || c >= R || a >= R) {
        return 0.0;
    }
    const Real r2 = R * R;
    const Real x_max = std::min(b, std::sqrt(std::max(0.0, r2 - c * c)));
    if (x_max <= a) {
        return 0.0;
    }

    const Real x_flat = (d < R) ? std::sqrt(std::max(0.0, r2 - d * d)) : 0.0;
    const Real x_d = std::clamp(x_flat, a, x_max);

    Real area = 0.0;
    if (x_d > a) {
        area += (x_d - a) * (d - c);
    }
    if (x_max > x_d) {
        auto antideriv = [R, r2](Real x) -> Real {
            const Real rad = std::sqrt(std::max(0.0, r2 - x * x));
            const Real arg = std::clamp(x / R, -1.0, 1.0);
            return 0.5 * (x * rad + r2 * std::asin(arg));
        };
        area += (antideriv(x_max) - antideriv(x_d)) - c * (x_max - x_d);
    }
    return area;
}

}  // namespace

Real disk_intersection_area_rect(Real x0, Real x1, Real y0, Real y1, Real radius) {
    if (x1 <= x0 || y1 <= y0 || radius <= 0.0) {
        return 0.0;
    }

    // Split [x0, x1] and [y0, y1] at 0 to map each piece into the first quadrant
    std::pair<Real, Real> xs[2];
    int nxs = 0;
    if (x0 < 0.0 && x1 > 0.0) {
        xs[nxs++] = {0.0, -x0};
        xs[nxs++] = {0.0, x1};
    } else if (x1 <= 0.0) {
        xs[nxs++] = {-x1, -x0};
    } else {
        xs[nxs++] = {x0, x1};
    }

    std::pair<Real, Real> ys[2];
    int nys = 0;
    if (y0 < 0.0 && y1 > 0.0) {
        ys[nys++] = {0.0, -y0};
        ys[nys++] = {0.0, y1};
    } else if (y1 <= 0.0) {
        ys[nys++] = {-y1, -y0};
    } else {
        ys[nys++] = {y0, y1};
    }

    Real total = 0.0;
    for (int ix = 0; ix < nxs; ++ix) {
        for (int iy = 0; iy < nys; ++iy) {
            total += disk_quadrant_rect_area(
                xs[ix].first, xs[ix].second, ys[iy].first, ys[iy].second, radius);
        }
    }
    return total;
}

}  // namespace fvm::geom
