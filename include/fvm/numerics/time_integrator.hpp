#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/types.hpp"

#include <stdexcept>
#include <vector>

namespace fvm::time {

enum class RKMethod {
    Euler,    // 1st-order Forward Euler
    SSP_RK2,  // 2nd-order Heun / TVD-RK2
    SSP_RK3,  // 3rd-order Shu-Osher TVD-RK3
    RK4       // Classical 4th-order Runge-Kutta
};

/// Effective Butcher weights b_s for each stage call inside RKStepper::step,
/// used to evaluate exact discrete conservation balance across boundary fluxes.
inline std::vector<Real> rk_stage_weights(RKMethod method) {
    switch (method) {
        case RKMethod::Euler:   return {1.0};
        case RKMethod::SSP_RK2: return {0.5, 0.5};
        case RKMethod::SSP_RK3: return {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
        case RKMethod::RK4:     return {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    }
    return {1.0};
}

namespace detail {

inline Real* raw_ptr(std::vector<Real>& v) { return v.data(); }
inline const Real* raw_ptr(const std::vector<Real>& v) { return v.data(); }
inline Index raw_size(const std::vector<Real>& v) { return v.size(); }

inline Real* raw_ptr(StructuredField2D& f) { return f.data(); }
inline const Real* raw_ptr(const StructuredField2D& f) { return f.data(); }
inline Index raw_size(const StructuredField2D& f) { return f.raw_size(); }

inline void match_shape(const std::vector<Real>& src, std::vector<Real>& dst) {
    dst.assign(src.size(), 0.0);
}

inline void match_shape(const StructuredField2D& src, StructuredField2D& dst) {
    if (dst.nx() != src.nx() || dst.ny() != src.ny() || dst.nghost() != src.nghost()) {
        dst = StructuredField2D(src.nx(), src.ny(), src.nghost(), 0.0);
    }
}

template <typename Field>
void copy_field(const Field& src, Field& dst) {
    match_shape(src, dst);
    const Index n = raw_size(src);
    const Real* s = raw_ptr(src);
    Real* d = raw_ptr(dst);
    for (Index i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

/// dst = a * x + b * y
template <typename Field>
void linear_comb2(Real a, const Field& x, Real b, const Field& y, Field& dst) {
    const Index n = raw_size(x);
    const Real* px = raw_ptr(x);
    const Real* py = raw_ptr(y);
    Real* pd = raw_ptr(dst);
    for (Index i = 0; i < n; ++i) {
        pd[i] = a * px[i] + b * py[i];
    }
}

/// dst = a * x + b * y + c * z
template <typename Field>
void linear_comb3(Real a, const Field& x, Real b, const Field& y,
                  Real c, const Field& z, Field& dst) {
    const Index n = raw_size(x);
    const Real* px = raw_ptr(x);
    const Real* py = raw_ptr(y);
    const Real* pz = raw_ptr(z);
    Real* pd = raw_ptr(dst);
    for (Index i = 0; i < n; ++i) {
        pd[i] = a * px[i] + b * py[i] + c * pz[i];
    }
}

}  // namespace detail

/// Allocation-reusing explicit Runge-Kutta stepper.
/// Contract for rhs(state, t_stage, dudt):
///   - rhs may enforce boundary conditions on `state` at `t_stage` (e.g. ghost cells),
///   - rhs must write d(state)/dt into `dudt` (leaving ghost cells 0 if StructuredField2D).
template <typename Field>
class RKStepper {
public:
    explicit RKStepper(RKMethod method = RKMethod::SSP_RK3) : method_(method) {}

    RKMethod method() const { return method_; }
    void set_method(RKMethod method) { method_ = method; }

    template <typename RhsFn>
    void step(Field& u, Real t, Real dt, RhsFn&& rhs) {
        ensure_workspace(u);
        switch (method_) {
            case RKMethod::Euler:
                step_euler(u, t, dt, rhs);
                break;
            case RKMethod::SSP_RK2:
                step_ssp_rk2(u, t, dt, rhs);
                break;
            case RKMethod::SSP_RK3:
                step_ssp_rk3(u, t, dt, rhs);
                break;
            case RKMethod::RK4:
                step_rk4(u, t, dt, rhs);
                break;
        }
    }

private:
    void ensure_workspace(const Field& u) {
        detail::match_shape(u, u1_);
        detail::match_shape(u, k1_);
        if (method_ == RKMethod::RK4) {
            detail::match_shape(u, k2_);
            detail::match_shape(u, k3_);
            detail::match_shape(u, k4_);
        }
    }

    template <typename RhsFn>
    void step_euler(Field& u, Real t, Real dt, RhsFn&& rhs) {
        rhs(u, t, k1_);
        detail::linear_comb2(1.0, u, dt, k1_, u);
    }

    template <typename RhsFn>
    void step_ssp_rk2(Field& u, Real t, Real dt, RhsFn&& rhs) {
        rhs(u, t, k1_);
        detail::linear_comb2(1.0, u, dt, k1_, u1_);
        rhs(u1_, t + dt, k1_);
        detail::linear_comb3(0.5, u, 0.5, u1_, 0.5 * dt, k1_, u);
    }

    template <typename RhsFn>
    void step_ssp_rk3(Field& u, Real t, Real dt, RhsFn&& rhs) {
        // Stage 1: u^(1) = u^n + dt * L(u^n, t)
        rhs(u, t, k1_);
        detail::linear_comb2(1.0, u, dt, k1_, u1_);

        // Stage 2: u^(2) = 3/4 u^n + 1/4 (u^(1) + dt * L(u^(1), t + dt))
        rhs(u1_, t + dt, k1_);
        detail::linear_comb3(0.75, u, 0.25, u1_, 0.25 * dt, k1_, u1_);

        // Stage 3: u^{n+1} = 1/3 u^n + 2/3 (u^(2) + dt * L(u^(2), t + dt/2))
        rhs(u1_, t + 0.5 * dt, k1_);
        detail::linear_comb3(1.0 / 3.0, u, 2.0 / 3.0, u1_, (2.0 / 3.0) * dt, k1_, u);
    }

    template <typename RhsFn>
    void step_rk4(Field& u, Real t, Real dt, RhsFn&& rhs) {
        rhs(u, t, k1_);
        detail::linear_comb2(1.0, u, 0.5 * dt, k1_, u1_);

        rhs(u1_, t + 0.5 * dt, k2_);
        detail::linear_comb2(1.0, u, 0.5 * dt, k2_, u1_);

        rhs(u1_, t + 0.5 * dt, k3_);
        detail::linear_comb2(1.0, u, dt, k3_, u1_);

        rhs(u1_, t + dt, k4_);

        const Index n = detail::raw_size(u);
        Real* pu = detail::raw_ptr(u);
        const Real* pk1 = detail::raw_ptr(k1_);
        const Real* pk2 = detail::raw_ptr(k2_);
        const Real* pk3 = detail::raw_ptr(k3_);
        const Real* pk4 = detail::raw_ptr(k4_);
        const Real c = dt / 6.0;
        for (Index i = 0; i < n; ++i) {
            pu[i] += c * (pk1[i] + 2.0 * pk2[i] + 2.0 * pk3[i] + pk4[i]);
        }
    }

    RKMethod method_;
    Field u1_;
    Field k1_;
    Field k2_;
    Field k3_;
    Field k4_;
};

}  // namespace fvm::time
