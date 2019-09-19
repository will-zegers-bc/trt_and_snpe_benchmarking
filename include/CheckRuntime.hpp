//==============================================================================
//
//  Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CHECKRUNTIME_H
#define CHECKRUNTIME_H

#include <string>

#include "SNPE/SNPEFactory.hpp"

namespace SNPE {

zdl::DlSystem::Runtime_t getRuntime(const std::string&);

} // namespace SNPE
#endif
