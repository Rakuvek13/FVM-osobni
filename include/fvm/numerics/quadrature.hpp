#pragma once

#include "fvm/core/types.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace fvm::quadrature {

struct QuadPoint1D {
    Real xi;      // in [0, 1]
    Real weight;  // sum(weight) = 1
};

struct QuadPointTri {
    Real lambda1;  // barycentric coord 1
    Real lambda2;  // barycentric coord 2
    Real lambda3;  // = 1 - lambda1 - lambda2
    Real weight;   // sum(weight) = 1
};

/// 1D Gauss-Legendre quadrature on [0, 1] for n in {1, 2, 3, 4, 5}.
/// Exact for polynomials up to degree 2n - 1.
inline std::vector<QuadPoint1D> gauss_legendre_01(int n) {
    auto map_m11_to_01 = [](Real x_m11, Real w_m11) -> QuadPoint1D {
        return {0.5 * (x_m11 + 1.0), 0.5 * w_m11};
    };

    switch (n) {
        case 1:
            return {{0.5, 1.0}};
        case 2: {
            const Real x = 1.0 / std::sqrt(3.0);
            return {map_m11_to_01(-x, 1.0), map_m11_to_01(x, 1.0)};
        }
        case 3: {
            const Real x = std::sqrt(3.0 / 5.0);
            const Real w_edge = 5.0 / 9.0;
            const Real w_mid = 8.0 / 9.0;
            return {
                map_m11_to_01(-x, w_edge),
                map_m11_to_01(0.0, w_mid),
                map_m11_to_01(x, w_edge)
            };
        }
        case 4: {
            const Real s = 2.0 * std::sqrt(6.0 / 5.0);
            const Real x1 = std::sqrt((3.0 - s) / 7.0);
            const Real x2 = std::sqrt((3.0 + s) / 7.0);
            const Real r30 = std::sqrt(30.0);
            const Real w1 = (18.0 + r30) / 36.0;
            const Real w2 = (18.0 - r30) / 36.0;
            return {
                map_m11_to_01(-x2, w2),
                map_m11_to_01(-x1, w1),
                map_m11_to_01(x1, w1),
                map_m11_to_01(x2, w2)
            };
        }
        case 5: {
            const Real s = 2.0 * std::sqrt(10.0 / 7.0);
            const Real x1 = std::sqrt(5.0 - s) / 3.0;
            const Real x2 = std::sqrt(5.0 + s) / 3.0;
            const Real r70 = std::sqrt(70.0);
            const Real w0 = 128.0 / 225.0;
            const Real w1 = (322.0 + 13.0 * r70) / 900.0;
            const Real w2 = (322.0 - 13.0 * r70) / 900.0;
            return {
                map_m11_to_01(-x2, w2),
                map_m11_to_01(-x1, w1),
                map_m11_to_01(0.0, w0),
                map_m11_to_01(x1, w1),
                map_m11_to_01(x2, w2)
            };
        }
        default:
            throw std::invalid_argument("gauss_legendre_01: supported n in {1, 2, 3, 4, 5}");
    }
}

/// 2D symmetric positive-weight triangle quadrature rules.
/// Exact for degrees: n=1 -> deg 1, n=3 -> deg 2, n=7 -> deg 5.
inline std::vector<QuadPointTri> triangle_rule(int min_degree) {
    if (min_degree <= 1) {
        return {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0}};
    }
    if (min_degree <= 2) {
        const Real a = 2.0 / 3.0;
        const Real b = 1.0 / 6.0;
        const Real w = 1.0 / 3.0;
        return {
            {a, b, b, w},
            {b, a, b, w},
            {b, b, a, w}
        };
    }
    if (min_degree <= 5) {
        // 7-point Radon / Hammer-Marlowe-Stroud rule (exact degree 5, strictly interior, all w > 0)
        const Real r15 = std::sqrt(15.0);
        const Real w0 = 9.0 / 40.0;
        const Real a1 = (9.0 + 2.0 * r15) / 21.0;
        const Real b1 = (6.0 - r15) / 21.0;
        const Real w1 = (155.0 - r15) / 1200.0;
        const Real a2 = (9.0 - 2.0 * r15) / 21.0;
        const Real b2 = (6.0 + r15) / 21.0;
        const Real w2 = (155.0 + r15) / 1200.0;
        return {
            {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, w0},
            {a1, b1, b1, w1},
            {b1, a1, b1, w1},
            {b1, b1, a1, w1},
            {a2, b2, b2, w2},
            {b2, a2, b2, w2},
            {b2, b2, a2, w2}
        };
    }
    throw std::invalid_argument("triangle_rule: supported min_degree <= 5");
}

}  // namespace fvm::quadrature
