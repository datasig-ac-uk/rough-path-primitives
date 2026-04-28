#ifndef RPP_UTILITY_HPP
#define RPP_UTILITY_HPP

#include <rpp/config.h>

namespace rpp {


template <typename... Args>
RPP_HOST_DEVICE
constexpr void ignore_unused(Args&&... arg RPP_MAYBE_UNUSED) noexcept {}

}

#endif // RPP_UTILITY_HPP
