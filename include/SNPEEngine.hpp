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
    std::vector<float>                      output;
    zdl::DlSystem::TensorMap                outputTensorMap;
    std::unique_ptr<zdl::SNPE::SNPE>        snpe;
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor;

public:
    SNPEEngine(const std::string&, const std::string&, const std::string&);
    std::vector<float> execute(pybind11::array_t<float, pybind11::array::c_style> input);
};
} // namespace SNPE
