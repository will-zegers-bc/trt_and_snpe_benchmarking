//==============================================================================
//
//  Copyright (c) 2015-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//
#include <string.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/StringList.hpp"
#include <pybind11/numpy.h>

#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "PerformanceProfile.hpp"
#include "SetBuilderOptions.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPEEngine.hpp"

namespace SNPE {

SNPEEngine::SNPEEngine(const std::string& dlc,
                       const std::string& runtimeString,
                       const std::string& performanceProfileString)
{
    std::ifstream dlcFileCheck(dlc);
    if (!dlcFileCheck)
    {
        throw std::runtime_error("DLC file not found");
    }

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    if (container == nullptr)
    {
        throw std::runtime_error("Error while opening the container file.");
    }

    zdl::DlSystem::Runtime_t runtime = getRuntime(runtimeString);
    zdl::DlSystem::PerformanceProfile_t performanceProfile = getPerformanceProfile(performanceProfileString);
    snpe = setBuilderOptions(container, runtime, performanceProfile);
    if (snpe == nullptr)
    {
        throw std::runtime_error("Error while building SNPE object.");
    }

    const auto &strList_opt = snpe->getInputTensorNames();
    const auto &strList = *strList_opt;

    auto inputShape = snpe->getInputDimensions(strList.at(0));
    inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
}

std::vector<float> SNPEEngine::execute(pybind11::array_t<float, pybind11::array::c_style> loadedFile)
{
    std::copy(loadedFile.data(), loadedFile.data() + loadedFile.size(), inputTensor->begin());
    if (!snpe->execute(inputTensor.get(), outputTensorMap))
    {
        throw std::runtime_error("Something went wrong when running inference");
    }

    zdl::DlSystem::StringList outputTensorNames = snpe->getOutputTensorNames();
    assert(outputTensorNames.size() == 1);
    auto outputTensor = outputTensorMap.getTensor(outputTensorNames.at(0));

    return std::vector<float>(outputTensor->begin(), outputTensor->end());
}
} // namespace SNPE
