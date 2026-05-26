#ifndef RPP_OPERATIONS_BASE_OPERATION_HPP
#define RPP_OPERATIONS_BASE_OPERATION_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/basis/basis_pack.hpp>

namespace rpp::ops {
template <typename Strategy_, typename = void>
class BaseOperation {
public:
    using Strategy = Strategy_;
    using Context = typename Strategy::Context;
    using Index = typename Strategy::Index;

    template <typename BasisPack>
    RPP_HOST static constexpr size_t
    scratch_space_size(Strategy const& strategy, BasisPack const& pack) noexcept {
        ignore_unused(strategy, pack);
        return 0;
    }

    template <typename Basis>
    RPP_HOST_DEVICE static void
    init_scratch_space(Context const& ctx, Basis const& basis) noexcept {
        ignore_unused(ctx, basis);
    }

    template <typename Basis>
    RPP_HOST_DEVICE static void
    destroy_scratch_space(Context const& ctx, Basis const& basis) noexcept {
        ignore_unused(ctx, basis);
    }
};

namespace detail {


template <typename Op,
          typename Context,
          typename ViewTuple,
          typename ExtrasTuple,
          size_t... Is,
          size_t... Js>
RPP_HOST_DEVICE constexpr void invoke_impl(Op const& op,
                                           Context const& ctx,
                                           ViewTuple&& views,
                                           ExtrasTuple&& extras,
                                           std::index_sequence<Is...>,
                                           std::index_sequence<Js...>) {
    op(ctx,
       std::get<Is>(views)...,
       std::get<Js>(std::forward<ExtrasTuple>(extras))...);
}
} // namespace detail

template <typename Op,
          typename Context,
          typename BatchMapper,
          typename BatchTuple,
          typename ExtrasTuple>
RPP_HOST_DEVICE constexpr void invoke(Op const& op,
                                      Context const& ctx,
                                      BatchMapper&& batch_mapper,
                                      BatchTuple const& batches,
                                      ExtrasTuple&& extras) {
    detail::invoke_impl(
        op,
        ctx,
        map_tuple(batches, std::forward<BatchMapper>(batch_mapper)),
        std::forward<ExtrasTuple>(extras),
        std::make_index_sequence<std::tuple_size_v<std::decay_t<BatchTuple>>>(),
        std::make_index_sequence<
            std::tuple_size_v<std::decay_t<ExtrasTuple>>>());
}
} // namespace rpp::ops

#endif // RPP_OPERATIONS_BASE_OPERATION_HPP
