#include "fvm/io/vtk_writer.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fvm::io {

namespace {

void ensure_parent_dir(const std::string& path) {
    const std::filesystem::path p(path);
    if (p.has_parent_path() && !p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path());
    }
}

}  // namespace

void write_vti(const std::string& path,
               const StructuredMesh2D& mesh,
               const std::vector<NamedScalarField>& scalars) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("write_vti: failed to open " + path);
    }

    for (const auto& s : scalars) {
        if (!s.field || s.field->nx() != mesh.nx() || s.field->ny() != mesh.ny()) {
            throw std::invalid_argument("write_vti: field dimensions do not match mesh");
        }
    }

    const std::string default_scalar = scalars.empty() ? "" : scalars.front().name;
    out << std::setprecision(15);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"0 " << mesh.nx() << " 0 " << mesh.ny() << " 0 0\""
        << " Origin=\"" << mesh.xmin() << " " << mesh.ymin() << " 0\""
        << " Spacing=\"" << mesh.dx() << " " << mesh.dy() << " 1\">\n";
    out << "    <Piece Extent=\"0 " << mesh.nx() << " 0 " << mesh.ny() << " 0 0\">\n";
    out << "      <CellData";
    if (!default_scalar.empty()) {
        out << " Scalars=\"" << default_scalar << "\"";
    }
    out << ">\n";

    for (const auto& s : scalars) {
        out << "        <DataArray type=\"Float64\" Name=\"" << s.name
            << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
        for (Index j = 0; j < mesh.ny(); ++j) {
            for (Index i = 0; i < mesh.nx(); ++i) {
                out << (*s.field)(static_cast<SignedIndex>(i), static_cast<SignedIndex>(j));
                if (i + 1 < mesh.nx()) out << ' ';
            }
            out << '\n';
        }
        out << "        </DataArray>\n";
    }

    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

PvdWriter::PvdWriter(std::string pvd_path) : pvd_path_(std::move(pvd_path)) {}

void PvdWriter::add_step(Real time, const std::string& relative_file) {
    entries_.emplace_back(time, relative_file);
}

void PvdWriter::save() const {
    ensure_parent_dir(pvd_path_);
    std::ofstream out(pvd_path_);
    if (!out) {
        throw std::runtime_error("PvdWriter::save: failed to open " + pvd_path_);
    }
    out << std::setprecision(15);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <Collection>\n";
    for (const auto& [time, file] : entries_) {
        out << "    <DataSet timestep=\"" << time << "\" part=\"0\" file=\"" << file << "\"/>\n";
    }
    out << "  </Collection>\n";
    out << "</VTKFile>\n";
}

}  // namespace fvm::io
