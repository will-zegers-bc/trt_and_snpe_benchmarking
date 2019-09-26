#pragma once

#include <string>
#include <vector>

#include <pybind11/numpy.h>

#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPE.hpp"

namespace SNPE {

class SNPEEngine
{
    size_t                           batchSize;
    zdl::DlSystem::Runtime_t         runtime;
    zdl::DlSystem::TensorShape       tensorShape;
    std::unique_ptr<zdl::SNPE::SNPE> snpe;

public:
    SNPEEngine(const std::string&, const std::string&);
    std::vector<float> execute(const std::string&);
    double measureLatency(pybind11::array_t<float, pybind11::array::c_style> input, int);
};
} // namespace SNPE
