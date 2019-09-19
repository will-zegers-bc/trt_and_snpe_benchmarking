//==============================================================================
//
//  Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <string.h>

#include <iostream>
#include <string>

#include "CheckRuntime.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/String.hpp"

namespace SNPE {

// Command line settings
zdl::DlSystem::Runtime_t getRuntime(const std::string& runtimeString)
{
    zdl::DlSystem::Runtime_t runtime;
    if (runtimeString.compare("gpu") == 0)
    {
        runtime = zdl::DlSystem::Runtime_t::GPU;
    }
    else if (runtimeString.compare("dsp") == 0)
    {
        runtime = zdl::DlSystem::Runtime_t::DSP;
    }
    else if (runtimeString.compare("cpu") == 0)
    {
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }
    else
    {
        throw std::runtime_error("Bad SNPE runtime string");
    }

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    return runtime;
}
} // namespace SNPE
