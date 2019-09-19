#pragma once

#include <string>
#include <vector>

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
    double measureLatency(const std::string&, size_t);
};
} // namespace SNPE
