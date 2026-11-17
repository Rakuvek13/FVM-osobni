#pragma once

#include "fvm/core/types.hpp"

#include <vector>

namespace fvm::geom {

/// Clips a 2D polygon against the half-plane a*x + b*y + c >= 0 (Sutherland-Hodgman).
std::vector<Vec2> clip_polygon_halfplane(const std::vector<Vec2>& poly, Real a, Real b, Real c);

/// Unsigned shoelace area of a simple 2D polygon.
Real polygon_area(const std::vector<Vec2>& poly);

/// Exact intersection area |poly \cap [xmin, xmax] x [ymin, ymax]|.
Real intersection_area_box(const std::vector<Vec2>& poly,
                           Real xmin, Real xmax, Real ymin, Real ymax);

/// Exact intersection area |poly \cap {|y - x| <= w}|.
Real intersection_area_diagonal_band(const std::vector<Vec2>& poly, Real w);

/// Exact 1D overlap fraction |[a, b] \cap [c, d]| / (b - a).
Real interval_overlap_fraction(Real a, Real b, Real c, Real d);

/// Exact analytical intersection area |[x0, x1] x [y0, y1] \cap {x^2 + y^2 <= R^2}|.
Real disk_intersection_area_rect(Real x0, Real x1, Real y0, Real y1, Real radius);

}  // namespace fvm::geom
