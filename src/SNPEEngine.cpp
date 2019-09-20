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
        std::cerr << "DLC file not found" << std::endl;
        throw std::runtime_error("DLC file not found");
    }
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    if (container == nullptr)
    {
        std::cerr << "Error while opening the container file." << std::endl;
        throw std::runtime_error("Error while opening the container file.");
    }
    snpe = setBuilderOptions(container, runtime);
    if (snpe == nullptr)
    {
        std::cerr << "Error while building SNPE object." << std::endl;
        throw std::runtime_error("Error while building SNPE object.");
    }
    tensorShape = snpe->getInputDimensions();
    batchSize = tensorShape.getDimensions()[0];
}

std::vector<float> SNPEEngine::execute(const std::string& inputFile)
{
    std::vector<float> output;
    zdl::DlSystem::TensorMap outputTensorMap;

    std::ifstream inputFileCheck(inputFile);
    if (!inputFileCheck)
    {
        std::cerr << "Input .raw file not found" << std::endl;
        throw std::runtime_error("Input file not found");
    }

    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputFile);

    if (snpe->execute(inputTensor.get(), outputTensorMap))
    {
        zdl::DlSystem::StringList outputTensorNames = snpe->getOutputTensorNames();
        assert(outputTensorNames.size() == 1);
        auto outputTensor = outputTensorMap.getTensor(outputTensorNames.at(0));

        output.assign(outputTensor->begin(), outputTensor->end());
    }
    else
    {
        throw std::runtime_error("Bad SNPE runtime string");
    }
    return output;
}

double SNPEEngine::measureLatency(const std::string& inputFile, int numRuns)
{
    std::vector<float> output;
    zdl::DlSystem::TensorMap outputTensorMap;

    std::ifstream inputFileCheck(inputFile);
    if (!inputFileCheck)
    {
        std::cerr << "Input .raw file not found" << std::endl;
        throw std::runtime_error("Input file not found");
    }

    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputFile);

    double avgTime = 0;
    for (int i = 0; i < numRuns; ++i)
    {
        auto t0 = std::chrono::steady_clock::now();

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

int main(int argc, char** argv)
{
    static std::string dlc = "";
    const char* inputFile = "";
    static std::string runtime = "cpu";

    int opt = 0;
    while ((opt = getopt(argc, argv, "hi:d:o:r")) != -1)
    {
        switch (opt)
        {
            case 'h':
                std::cout
                        << "\nDESCRIPTION:\n"
                        << "------------\n"
                        << "Example application demonstrating how to load and execute a neural network\n"
                        << "using the SNPE C++ API.\n"
                        << "\n\n"
                        << "REQUIRED ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -d  <FILE>   Path to the DL container containing the network.\n"
                        << "  -i  <FILE>   Path to a file listing the inputs for the network.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -r  <RUNTIME> The runtime to be used [gpu, dsp, cpu] (cpu is default). \n"
                        << std::endl;

                std::exit(SUCCESS);
            case 'i':
                inputFile = optarg;
                break;
            case 'd':
                dlc = optarg;
                break;
            case 'r':
                runtime = optarg;
                break;
            default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments" << std::endl;
                std::exit(FAILURE);
        }
    }

    SNPE::SNPEEngine snpe(dlc, runtime);
    auto output = snpe.execute(inputFile);

    return SUCCESS;
}
