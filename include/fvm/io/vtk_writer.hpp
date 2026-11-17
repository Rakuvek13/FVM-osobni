#pragma once

#include "fvm/core/field.hpp"
#include "fvm/core/types.hpp"
#include "fvm/mesh/structured_mesh.hpp"
#include "fvm/mesh/unstructured_mesh.hpp"

#include <string>
#include <utility>
#include <vector>

namespace fvm::io {

struct NamedScalarField {
    std::string name;
    const StructuredField2D* field = nullptr;
};

struct UnstructuredScalarField {
    std::string name;
    const std::vector<Real>* data = nullptr;
};

struct UnstructuredVectorField {
    std::string name;
    const std::vector<Vec2>* data = nullptr;
};

/// Writes 2D structured cell-centered fields to VTK XML ImageData (.vti) for ParaView.
void write_vti(const std::string& path,
               const StructuredMesh2D& mesh,
               const std::vector<NamedScalarField>& scalars);

/// Writes 2D unstructured cell-centered scalar/vector fields to VTK XML UnstructuredGrid (.vtu).
void write_vtu(const std::string& path,
               const UnstructuredMesh2D& mesh,
               const std::vector<UnstructuredScalarField>& scalars,
               const std::vector<UnstructuredVectorField>& vectors = {});

/// Writes a ParaView Data (.pvd) collection file indexing time-series .vti/.vtu snapshots.
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
