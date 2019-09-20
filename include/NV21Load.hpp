//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef NV21LOAD_H
#define NV21LOAD_H
#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensorFactory.hpp"

#include "DlSystem/TensorMap.hpp"

namespace SNPE {

std::unique_ptr<zdl::DlSystem::ITensor> loadNV21Tensor (std::unique_ptr<zdl::SNPE::SNPE> & snpe , const char* inputFileListPath);

} // namespace SNPE
#endif