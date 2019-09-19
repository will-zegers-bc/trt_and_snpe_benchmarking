//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOADCONTAINER_H
#define LOADCONTAINER_H

#include <string>

#include "DlContainer/IDlContainer.hpp"

namespace SNPE {

std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath);

}
#endif
