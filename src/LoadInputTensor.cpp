//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

#include "LoadInputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

namespace SNPE {

// Load a batched single input tensor for a network which requires a single input
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE>& snpe, const std::string& filePath)
{
    std::unique_ptr<zdl::DlSystem::ITensor> input;
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;

    assert (strList.size() == 1);

    std::vector<float> loadedFile = loadFloatDataFile(filePath);

    const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;

    input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

    if (input->getSize() != loadedFile.size()) {
        std::cerr << "Size of input does not match network.\n"
                  << "Expecting: " << input->getSize() << "\n"
                  << "Got: " << loadedFile.size() << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::copy(loadedFile.begin(), loadedFile.end(), input->begin());
    return std::move(input);
}

// Load multiple input tensors for a network which require multiple inputs
zdl::DlSystem::TensorMap loadMultipleInput (std::unique_ptr<zdl::SNPE::SNPE>& snpe , std::vector<std::string>& fileLines)
{
    const auto& inputTensorNamesRef = snpe->getInputTensorNames();
    if (!inputTensorNamesRef) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &inputTensorNames = *inputTensorNamesRef;
    // Make sure the network requires multiple inputs
    assert (inputTensorNames.size() > 1);

    if (inputTensorNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputs(inputTensorNames.size());
    zdl::DlSystem::TensorMap  inputTensorMap;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // Treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        for (size_t j = 0; j<inputTensorNames.size(); j++) {

            // print out which file is being processed
            std::string filePath(filePaths[j]);
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            std::string inputName(inputTensorNames.at(j));
            std::vector<float> inputVec = loadFloatDataFile(filePath);

            const auto &inputShape_opt = snpe->getInputDimensions(inputTensorNames.at(j));
            const auto &inputShape = *inputShape_opt;
            inputs[j] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

            if (inputs[j]->getSize() != inputVec.size()) {
                std::cerr << "Size of input does not match network.\n"
                          << "Expecting: " << inputs[j]->getSize() << "\n"
                          << "Got: " << inputVec.size() << "\n";
                std::exit(EXIT_FAILURE);
            }

            std::copy(inputVec.begin(), inputVec.end(), inputs[j]->begin());
            inputTensorMap.add(inputName.c_str(), inputs[j].release());
        }
    }
    std::cout << "Finished processing inputs for current inference \n";
    return inputTensorMap;
}

void loadInputUserBufferTf8(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines,
                         zdl::DlSystem::UserBufferMap& inputMap)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        for (size_t j = 0; j < inputNames.size(); j++) {
            const char *name = inputNames.at(j);
            std::string filePath(filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            unsigned char stepEquivalentTo0;
            float quantizedStepSize;
            loadByteDataFileBatchedTf8(filePath, applicationBuffers.at(name), i, stepEquivalentTo0, quantizedStepSize);
            auto userBufferEncoding = dynamic_cast<zdl::DlSystem::UserBufferEncodingTf8 *>(&inputMap.getUserBuffer(name)->getEncoding());
            userBufferEncoding->setStepExactly0(stepEquivalentTo0);
            userBufferEncoding->setQuantizedStepSize(quantizedStepSize);

        }
    }
}

// Load multiple batched input user buffers
void loadInputUserBufferFloat(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    if (inputNames.size()) std::cout << "Processing DNN Input: " << std::endl;

    for(size_t i=0; i<fileLines.size(); i++) {
        std::string fileLine(fileLines[i]);
        // treat each line as a space-separated list of input files
        std::vector<std::string> filePaths;
        split(filePaths, fileLine, ' ');

        for (size_t j = 0; j < inputNames.size(); j++) {
            const char *name = inputNames.at(j);
            std::string filePath(filePaths[j]);

            // print out which file is being processed
            std::cout << "\t" << j + 1 << ") " << filePath << std::endl;

            // load file content onto application storage buffer,
            // on top of which, SNPE has created a user buffer
            loadByteDataFileBatched(filePath, applicationBuffers.at(name), i);
        }
    }
}

void loadInputUserBuffer(std::unordered_map<std::string, GLuint>& applicationBuffers,
                               std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                               const GLuint inputglbuffer)
{
    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    assert(inputNames.size() > 0);

    for (size_t i = 0; i < inputNames.size(); i++) {
        const char* name = inputNames.at(i);
        applicationBuffers.at(name) = inputglbuffer;
    };
}
} // namespace SNPE
