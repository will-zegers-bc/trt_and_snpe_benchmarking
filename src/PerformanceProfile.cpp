#include <string>

#include "PerformanceProfile.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"

namespace SNPE
{

zdl::DlSystem::PerformanceProfile_t getPerformanceProfile(const std::string& performanceProfileString)
{
    if (performanceProfileString.compare("balanced") == 0)
    {
        return zdl::DlSystem::PerformanceProfile_t::BALANCED;
    }
    else if (performanceProfileString.compare("high_performance") == 0)
    {
        return zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;
    }
    else if (performanceProfileString.compare("power_saver") == 0)
    {
        return zdl::DlSystem::PerformanceProfile_t::POWER_SAVER;
    }
    else if (performanceProfileString.compare("system_settings") == 0)
    {
        return zdl::DlSystem::PerformanceProfile_t::SYSTEM_SETTINGS;
    }
    else if (performanceProfileString.compare("sustained_high_performance") == 0)
    {
        return zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE;
    }
    else if (performanceProfileString.compare("burst") == 0)
    {
        return zdl::DlSystem::PerformanceProfile_t::BURST;
    }
    else
    {
        return zdl::DlSystem::PerformanceProfile_t::DEFAULT;
    }
}
}
