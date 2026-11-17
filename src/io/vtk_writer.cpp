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

int vtk_cell_type(Index num_vertices) {
    if (num_vertices == 3) return 5;  // VTK_TRIANGLE
    if (num_vertices == 4) return 9;  // VTK_QUAD
    return 7;                         // VTK_POLYGON
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

void write_vtu(const std::string& path,
               const UnstructuredMesh2D& mesh,
               const std::vector<UnstructuredScalarField>& scalars,
               const std::vector<UnstructuredVectorField>& vectors) {
    ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("write_vtu: failed to open " + path);
    }

    const Index n_cells = mesh.num_cells();
    for (const auto& s : scalars) {
        if (!s.data || s.data->size() != n_cells) {
            throw std::invalid_argument("write_vtu: scalar field size mismatch");
        }
    }
    for (const auto& v : vectors) {
        if (!v.data || v.data->size() != n_cells) {
            throw std::invalid_argument("write_vtu: vector field size mismatch");
        }
    }

    out << std::setprecision(15);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << mesh.num_vertices()
        << "\" NumberOfCells=\"" << n_cells << "\">\n";

    // Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& p : mesh.vertices()) {
        out << "          " << p.x << ' ' << p.y << " 0\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Cells
    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& cell : mesh.cells()) {
        out << "         ";
        for (Index vid : cell.vertex_ids) {
            out << ' ' << vid;
        }
        out << '\n';
    }
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n         ";
    Index offset = 0;
    for (const auto& cell : mesh.cells()) {
        offset += cell.vertex_ids.size();
        out << ' ' << offset;
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n         ";
    for (const auto& cell : mesh.cells()) {
        out << ' ' << vtk_cell_type(cell.vertex_ids.size());
    }
    out << "\n        </DataArray>\n";
    out << "      </Cells>\n";

    // CellData
    out << "      <CellData";
    if (!scalars.empty()) {
        out << " Scalars=\"" << scalars.front().name << "\"";
    }
    if (!vectors.empty()) {
        out << " Vectors=\"" << vectors.front().name << "\"";
    }
    out << ">\n";

    for (const auto& s : scalars) {
        out << "        <DataArray type=\"Float64\" Name=\"" << s.name
            << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
        for (Index c = 0; c < n_cells; ++c) {
            out << (*s.data)[c] << ((c + 1) % 8 == 0 || c + 1 == n_cells ? '\n' : ' ');
        }
        out << "        </DataArray>\n";
    }

    for (const auto& v : vectors) {
        out << "        <DataArray type=\"Float64\" Name=\"" << v.name
            << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (Index c = 0; c < n_cells; ++c) {
            out << "          " << (*v.data)[c].x << ' ' << (*v.data)[c].y << " 0\n";
        }
        out << "        </DataArray>\n";
    }

    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
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
