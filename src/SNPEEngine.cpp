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

#include <pybind11/numpy.h>

#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "PreprocessInput.hpp"
#include "Util.hpp"
#include "SNPEEngine.hpp"

#include "DlSystem/StringList.hpp"
#include "SNPE/SNPE.hpp"

const int FAILURE = 1;
const int SUCCESS = 0;

namespace SNPE {

SNPEEngine::SNPEEngine(const std::string& dlc, const std::string& runtimeString)
    : runtime(getRuntime(runtimeString))
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
    snpe = setBuilderOptions(container, runtime);
    if (snpe == nullptr)
    {
        throw std::runtime_error("Error while building SNPE object.");
    }
    const auto &strList_opt = snpe->getInputTensorNames();
    const auto &strList = *strList_opt;

    inputShape = snpe->getInputDimensions(strList.at(0));
    inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
}

std::vector<float> SNPEEngine::execute(pybind11::array_t<float, pybind11::array::c_style> loadedFile)
{
    std::copy(loadedFile.data(), loadedFile.data() + loadedFile.size(), inputTensor->begin());
    if (snpe->execute(inputTensor.get(), outputTensorMap))
    {
        zdl::DlSystem::StringList outputTensorNames = snpe->getOutputTensorNames();
        assert(outputTensorNames.size() == 1);
        auto outputTensor = outputTensorMap.getTensor(outputTensorNames.at(0));

        output.assign(outputTensor->begin(), outputTensor->end());
    }
    else
    {
        throw std::runtime_error("Something went wrong when running inference");
    }
    return output;
}

double SNPEEngine::measureLatency(pybind11::array_t<float, pybind11::array::c_style> loadedFile, int numRuns)
{
    double avgTime = 0;
    for (int i = 0; i < numRuns; ++i)
    {
        auto t0 = std::chrono::steady_clock::now();
        std::copy(loadedFile.data(), loadedFile.data() + loadedFile.size(), inputTensor->begin());

        if (snpe->execute(inputTensor.get(), outputTensorMap))
        {
            zdl::DlSystem::StringList outputTensorNames = snpe->getOutputTensorNames();
            assert(outputTensorNames.size() == 1);
            auto outputTensor = outputTensorMap.getTensor(outputTensorNames.at(0));

            output.assign(outputTensor->begin(), outputTensor->end());
        }
        else
        {
            throw std::runtime_error("Something went wrong when running inference");
        }

        std::chrono::duration<double> diff = std::chrono::steady_clock::now() - t0;
        if (i != 0)
        {
          avgTime += diff.count();
        }
    }
    return avgTime / numRuns;
}
} // namespace SNPE
