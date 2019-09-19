//==============================================================================
//
//  Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOADINPUTTENSOR_H
#define LOADINPUTTENSOR_H

#include <unordered_map>
#include <string>
#include <vector>

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorMap.hpp"

namespace SNPE {

typedef unsigned int GLuint;
std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE>& snpe , const std::string& filePath);
zdl::DlSystem::TensorMap loadMultipleInput (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::string& fileLine);

void loadInputUserBufferFloat(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                                std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                                std::vector<std::string>& fileLines);

void loadInputUserBufferTf8(std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                         std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                         std::vector<std::string>& fileLines,
                         zdl::DlSystem::UserBufferMap& inputMap);

void loadInputUserBuffer(std::unordered_map<std::string, GLuint>& applicationBuffers,
                                std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                                const GLuint inputglbuffer);
} // namespace SNPE
#endif
