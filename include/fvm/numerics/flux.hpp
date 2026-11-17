#pragma once

#include "fvm/core/types.hpp"

#include <algorithm>
#include <cmath>

namespace fvm::flux {

/// Exact upwind flux for linear advection f(u) = v_n * u across a face with
/// outward normal speed v_n = v . n (from Left to Right).
inline Real linear_upwind(Real u_left, Real u_right, Real v_n) {
    return (v_n >= 0.0) ? (v_n * u_left) : (v_n * u_right);
}

/// Scalar Rusanov / Local Lax-Friedrichs numerical flux:
///   F_hat(u_L, u_R) = 0.5 * (f_L + f_R) - 0.5 * alpha * (u_R - u_L)
/// where alpha >= max_{u in [min(u_L,u_R), max(u_L,u_R)]} |f'(u)|.
inline Real rusanov(Real f_left, Real f_right, Real u_left, Real u_right, Real alpha) {
    return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left);
}

/// Safe interval maximum |f'(u)| over [min(u_L, u_R), max(u_L, u_R)] via uniform sampling.
/// Essential for nonconvex fluxes (e.g. KPP) where max(|f'(u_L)|, |f'(u_R)|) can miss
/// interior extrema of |f'(u)|!
template <typename DerivFn>
Real interval_max_wave_speed(Real u_left, Real u_right, DerivFn&& df, int samples = 16) {
    const Real u_min = std::min(u_left, u_right);
    const Real u_max = std::max(u_left, u_right);
    Real alpha = std::max(std::abs(df(u_min)), std::abs(df(u_max)));
    if (u_max == u_min || samples <= 1) {
        return alpha;
    }
    const Real du = (u_max - u_min) / static_cast<Real>(samples);
    for (int k = 1; k < samples; ++k) {
        const Real u = u_min + static_cast<Real>(k) * du;
        alpha = std::max(alpha, std::abs(df(u)));
    }
    return alpha;
}

}  // namespace fvm::flux
