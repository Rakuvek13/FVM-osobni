#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/types.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fvm {

struct ConvergenceRow {
    std::string scheme;
    Index num_cells = 0;
    Real h = 0.0;
    ErrorNorms errors;
    Real eoc_l1 = 0.0;
    Real eoc_l2 = 0.0;
    Real eoc_linf = 0.0;
    Real mass_drift = 0.0;
    Real runtime_sec = 0.0;
};

inline Real experimental_order(Real err_coarse, Real err_fine, Real h_coarse, Real h_fine) {
    if (err_coarse <= 0.0 || err_fine <= 0.0 || h_coarse <= h_fine || h_fine <= 0.0) {
        return 0.0;
    }
    return std::log(err_coarse / err_fine) / std::log(h_coarse / h_fine);
}

inline void populate_eoc(std::vector<ConvergenceRow>& rows) {
    for (Index i = 1; i < rows.size(); ++i) {
        if (rows[i].scheme != rows[i - 1].scheme) {
            continue;
        }
        rows[i].eoc_l1 = experimental_order(
            rows[i - 1].errors.l1, rows[i].errors.l1, rows[i - 1].h, rows[i].h);
        rows[i].eoc_l2 = experimental_order(
            rows[i - 1].errors.l2, rows[i].errors.l2, rows[i - 1].h, rows[i].h);
        rows[i].eoc_linf = experimental_order(
            rows[i - 1].errors.linf, rows[i].errors.linf, rows[i - 1].h, rows[i].h);
    }
}

inline void print_convergence_table(std::ostream& os, const std::vector<ConvergenceRow>& rows) {
    os << std::left << std::setw(14) << "Scheme"
       << std::right << std::setw(10) << "Cells"
       << std::setw(12) << "h"
       << std::setw(13) << "L1"
       << std::setw(8) << "EOC_1"
       << std::setw(13) << "L2"
       << std::setw(8) << "EOC_2"
       << std::setw(13) << "Linf"
       << std::setw(8) << "EOC_inf"
       << std::setw(12) << "MassDrift"
       << std::setw(10) << "Time(s)" << '\n';
    os << std::string(121, '-') << '\n';

    for (const auto& r : rows) {
        os << std::left << std::setw(14) << r.scheme
           << std::right << std::setw(10) << r.num_cells
           << std::scientific << std::setprecision(3) << std::setw(12) << r.h
           << std::setw(13) << r.errors.l1
           << std::fixed << std::setprecision(2) << std::setw(8) << r.eoc_l1
           << std::scientific << std::setprecision(3) << std::setw(13) << r.errors.l2
           << std::fixed << std::setprecision(2) << std::setw(8) << r.eoc_l2
           << std::scientific << std::setprecision(3) << std::setw(13) << r.errors.linf
           << std::fixed << std::setprecision(2) << std::setw(8) << r.eoc_linf
           << std::scientific << std::setprecision(2) << std::setw(12) << r.mass_drift
           << std::fixed << std::setprecision(3) << std::setw(10) << r.runtime_sec << '\n';
    }
}

inline void write_convergence_csv(const std::string& path,
                                  const std::vector<ConvergenceRow>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV output file: " + path);
    }
    out << "scheme,num_cells,h,l1,eoc_l1,l2,eoc_l2,linf,eoc_linf,mass_drift,runtime_sec\n";
    out << std::setprecision(15);
    for (const auto& r : rows) {
        out << r.scheme << ','
            << r.num_cells << ','
            << r.h << ','
            << r.errors.l1 << ','
            << r.eoc_l1 << ','
            << r.errors.l2 << ','
            << r.eoc_l2 << ','
            << r.errors.linf << ','
            << r.eoc_linf << ','
            << r.mass_drift << ','
            << r.runtime_sec << '\n';
    }
}

}  // namespace fvm
