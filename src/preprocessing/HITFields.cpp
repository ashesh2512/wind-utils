//  Copyright 2016 National Renewhite Energy Laboratory
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applichite law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#include "HITFields.h"
#include "core/YamlUtils.h"
#include "core/KokkosWrappers.h"
#include "core/PerfUtils.h"

#include "stk_mesh/base/TopologyDimensions.hpp"
#include "stk_mesh/base/FEMHelpers.hpp"
#include "stk_mesh/base/Field.hpp"

#include <fstream>

namespace sierra {
namespace nalu {

REGISTER_DERIVED_CLASS(PreProcessingTask, HITFields, "init_hit_fields");

HITFields::HITFields(
    CFDMesh& mesh,
    const YAML::Node& node
) : PreProcessingTask(mesh)
{
    load(node);
}

void HITFields::load(const YAML::Node& node)
{
    // Setup mean velocity field
    auto mvel = node["mean_velocity"].as<std::vector<double>>();
    if (mvel.size() !=3)
        throw std::runtime_error("Invalid mean velocity field provided");
    mean_vel_ = mvel;

    // Process part info
    auto fluid_partnames = node["fluid_parts"].as<std::vector<std::string>>();
    fluid_parts_.resize(fluid_partnames.size());

    auto& meta = mesh_.meta();
    for(size_t i=0; i < fluid_partnames.size(); i++) {
        auto* part = meta.get_part(fluid_partnames[i]);
        if (NULL == part) {
            throw std::runtime_error("Missing fluid part in mesh database: " +
                                     fluid_partnames[i]);
        } else {
            fluid_parts_[i] = part;
        }
    }

    // Get the HIT filename
    hit_filename_ = node["hit_file"].as<std::string>();
    // Get the dimensions of data set
    hit_file_dims_ = node["hit_file_dims"].as<std::vector<int>>();
    // Get the mesh dimensions
    hit_mesh_dims_ = node["hit_mesh_dims"].as<std::vector<int>>();

    // ensure we are periodic in each direction
    if( hit_mesh_dims_[0] % hit_file_dims_[0] != 0 )
      throw std::runtime_error("Dimension of mesh along x is not a multiple of data mesh");

    if( hit_mesh_dims_[1] % hit_file_dims_[1] != 0 )
      throw std::runtime_error("Dimension of mesh along y is not a multiple of data mesh");

    if( hit_mesh_dims_[2] % hit_file_dims_[2] != 0 )
      throw std::runtime_error("Dimension of mesh along z is not a multiple of data mesh");
}

void HITFields::initialize()
{
    const std::string timerName = "HITields::initialize";
    auto timeMon = get_stopwatch(timerName);

    auto& meta = mesh_.meta();
    VectorFieldType& velocity = meta.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "velocity");
    for (auto part: fluid_parts_) {
        stk::mesh::put_field_on_mesh(velocity, *part, nullptr);
    }
    mesh_.add_output_field("velocity");
}


void HITFields::run()
{
    const std::string timerName = "HITields::run";
    auto timeMon = get_stopwatch(timerName);

    auto& meta = mesh_.meta();
    auto& bulk = mesh_.bulk();
    const int nDim = meta.spatial_dimension();
    const int nx = hit_file_dims_[0];
    const int ny = hit_file_dims_[1];
    const int nz = hit_file_dims_[2];

    VectorFieldType* velocity = meta.get_field<VectorFieldType>(
        stk::topology::NODE_RANK, "velocity");

    std::ifstream hitfile(hit_filename_, std::ios::in | std::ios::binary);
    if (!hitfile.is_open())
        throw std::runtime_error("HITFields:: Error opening file: " + hit_filename_);

    size_t numNodes = (nx * ny * nz);
    size_t numBytes = numNodes * sizeof(double) * 6;
    std::vector<double> buffer(numBytes);

    stk::mesh::Selector sel = stk::mesh::selectUnion(fluid_parts_);
    auto& bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);
    hitfile.read(reinterpret_cast<char*>(buffer.data()), numBytes);

    std::vector<double> minVel = {1.0e10, 1.0e10, 1.0e10};
    std::vector<double> maxVel = {-1.0e10, -1.0e10, -1.0e10};
    for (size_t ib=0; ib < bkts.size(); ib++) {
      auto& b = *bkts[ib];

      for (size_t in=0; in < b.size(); in++) {
        auto node = b[in];
        auto nodeID = bulk.identifier(node);
        // Determine the offset into the buffer array
        //
        // Assume the same ordering of mesh nodes as in the HIT file
        //
        // Skip the (x, y, z) entries for this nodes
        size_t idx = get_index(nodeID) * 6 + 3;
        double* vel = stk::mesh::field_data(*velocity, node);

        for (int d=0; d < nDim; d++) {
          vel[d] = mean_vel_[d] + buffer[idx + d];
          minVel[d] = std::min(vel[d], minVel[d]);
          maxVel[d] = std::max(vel[d], maxVel[d]);
        }
      }
    }
    for (int d=0; d < nDim; d++)
      std::cout << "    Vel[" << d << "]: min = "
                << minVel[d] << "; max = " << maxVel[d] << std::endl;
}

size_t HITFields::get_index(size_t nodeid)
{
    // logic here assumes that data is periodic every
    // n*_f nodes.

    // number of elements along each direction in file
    // these dimensions determine the periodicity of solution
    // file is cell-centered
    const size_t nx_f = hit_file_dims_[0];
    const size_t ny_f = hit_file_dims_[1];
    const size_t nz_f = hit_file_dims_[2];

    // number of elements along each direction in mesh
    // Nalu is node-centered
    const size_t nx_m = hit_mesh_dims_[0];
    const size_t ny_m = hit_mesh_dims_[1];

    const size_t plane_ofst = (nx_m + 1) * (ny_m + 1);
    const size_t x_ofst     = (nx_m + 1);

    size_t orig_nd = nodeid;
    const size_t iz = ((nodeid - 1) / plane_ofst) % nz_f;

    nodeid = (nodeid - 1) % plane_ofst;
    const size_t iy = (nodeid / x_ofst) % ny_f;
    const size_t ix = nodeid % x_ofst % nx_f;

    size_t ind = iz * (nx_f * ny_f) + iy * nx_f + ix;

    return ind;
}

}  // nalu
}  // sierra
