#ifndef RPP_OPERATIONS_BASE_OPERATION_HPP
#define RPP_OPERATIONS_BASE_OPERATION_HPP

#include <algorithm>
#include <cstddef>
#include <utility>
#include <type_traits>

#include <rpp/config.h>
#include <rpp/utility.hpp>

namespace rpp::ops {
template<typename Strategy, typename=void>
class BaseOperation {
public:
    using Context = typename Strategy::Context;

    template<typename Basis>
    RPP_HOST
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template<typename Basis>
    RPP_HOST_DEVICE
    static void init_scratch_space(Context const &ctx, Basis const &basis) noexcept {
        ignore_unused(ctx, basis);
    }

    template<typename Basis>
    RPP_HOST_DEVICE
    static void destroy_scratch_space(Context const &ctx, Basis const &basis) noexcept {
        ignore_unused(ctx, basis);
    }
};

namespace detail {
template<typename Op, typename Context, typename BatchMapper, typename BatchTuple, typename ExtrasTuple, size_t... Is,
    size_t... Js>
void invoke_impl(Op const &op, Context const &ctx, BatchMapper &&batch_mapper, BatchTuple const &batches,
                 ExtrasTuple extras, std::index_sequence<Is...>, std::index_sequence<Js...>) {
    op(ctx, batch_mapper(std::get<Is>(batches))..., std::get<Js>(extras)...);
}
}

template<typename Op, typename Context, typename BatchMapper, typename BatchTuple, typename ExtrasTuple>
void invoke(Op const &op, Context const &ctx, BatchMapper &&batch_mapper, BatchTuple const &batches,
            ExtrasTuple extras) {
    detail::invoke_impl(op, ctx, std::forward<BatchMapper>(batch_mapper), batches, extras,
                        std::make_index_sequence<std::tuple_size_v<BatchTuple> >(),
                        std::make_index_sequence<std::tuple_size_v<ExtrasTuple> >()
                        );
}
} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASE_OPERATION_HPP
