#ifndef RPP_OPERATIONS_BASE_OPERATION_HPP
#define RPP_OPERATIONS_BASE_OPERATION_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class BaseOperation {
public:
    using Context = typename Strategy::Context;

    template <typename Basis>
    RPP_HOST
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename Basis>
    RPP_HOST_DEVICE
    static void init_scratch_space(Context const& ctx, Basis const& basis) noexcept {
        ignore_unused(ctx, basis);
    }

    template <typename Basis>
    RPP_HOST_DEVICE
    static void destroy_scratch_space(Context const& ctx, Basis const& basis) noexcept {
        ignore_unused(ctx, basis);
    }
};


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASE_OPERATION_HPP
