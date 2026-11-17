#pragma once

#include <cmath>
#include <cstddef>

namespace fvm {

using Real = double;
using Index = std::size_t;
using SignedIndex = std::ptrdiff_t;

inline constexpr Real PI = 3.141592653589793238462643383279502884;
inline constexpr Real EPS = 1.0e-14;

struct Vec2 {
    Real x = 0.0;
    Real y = 0.0;

    constexpr Vec2() = default;
    constexpr Vec2(Real x_, Real y_) : x(x_), y(y_) {}

    constexpr Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    constexpr Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    constexpr Vec2 operator-() const { return {-x, -y}; }
    constexpr Vec2 operator*(Real s) const { return {x * s, y * s}; }
    constexpr Vec2 operator/(Real s) const { return {x / s, y / s}; }

    constexpr Vec2& operator+=(const Vec2& o) {
        x += o.x;
        y += o.y;
        return *this;
    }
    constexpr Vec2& operator-=(const Vec2& o) {
        x -= o.x;
        y -= o.y;
        return *this;
    }
    constexpr Vec2& operator*=(Real s) {
        x *= s;
        y *= s;
        return *this;
    }

    constexpr Real norm2() const { return x * x + y * y; }
    Real norm() const { return std::sqrt(norm2()); }

    Vec2 normalized() const {
        const Real n = norm();
        return n > 0.0 ? Vec2{x / n, y / n} : Vec2{0.0, 0.0};
    }

    /// Outward normal direction for a counter-clockwise oriented edge (v0 -> v1).
    constexpr Vec2 perp_cw() const { return {y, -x}; }

    /// Left-hand normal direction for an edge (v0 -> v1).
    constexpr Vec2 perp_ccw() const { return {-y, x}; }
};

constexpr Vec2 operator*(Real s, const Vec2& v) { return {s * v.x, s * v.y}; }
constexpr Real dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
constexpr Real cross(const Vec2& a, const Vec2& b) { return a.x * b.y - a.y * b.x; }

inline Real orient2d(const Vec2& a, const Vec2& b, const Vec2& c) {
    return cross(b - a, c - a);
}

}  // namespace fvm
