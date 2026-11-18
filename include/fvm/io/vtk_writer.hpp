#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/structured_mesh.hpp"

#include <string>
#include <utility>
#include <vector>

namespace fvm::io {

struct NamedScalarField {
    std::string name;
    const StructuredField2D* field = nullptr;
};

/// Writes 2D structured cell-centered fields to VTK XML ImageData (.vti) for ParaView.
void write_vti(const std::string& path,
               const StructuredMesh2D& mesh,
               const std::vector<NamedScalarField>& scalars);

/// Writes a ParaView Data (.pvd) collection file indexing time-series .vti snapshots.
class PvdWriter {
public:
    explicit PvdWriter(std::string pvd_path);

    void add_step(Real time, const std::string& relative_file);
    void save() const;

    const std::string& path() const { return pvd_path_; }
    Index size() const { return entries_.size(); }

private:
    std::string pvd_path_;
    std::vector<std::pair<Real, std::string>> entries_;
};

}  // namespace fvm::io
