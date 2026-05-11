#ifndef RPP_GPU_OPERATIONS_BLOCK_KERNEL_HPP
#define RPP_GPU_OPERATIONS_BLOCK_KERNEL_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>

#include <rpp/basis/basis_pack.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>

namespace rpp::gpu::block {
namespace detail {
template<typename Op, typename Context, typename BatchMapper, typename BatchTuple, typename... Extras, size_t... Is>
void invoke_impl(Op const &op, Context const &ctx, BatchMapper &&batch_mapper, BatchTuple const &batches,
                 Extras &&... extras, std::index_sequence<Is...>) {
    op(ctx, batch_mapper(std::get<Is>(batches))..., std::forward<Extras>(extras)...);
}


template<typename Op, typename Context, typename BatchMapper, typename BatchTuple, typename... Extras>
void invoke(Op const &op, Context const &ctx, BatchMapper &&batch_mapper, BatchTuple const &batches,
            Extras &&... extras) {
    invoke_impl(op, ctx, batch_mapper, batches, extras..., std::make_index_sequence<std::tuple_size_v<BatchTuple> >());
}
} // namespace detail

template<
    typename Op,
    typename... Bases,
    typename... BatchArgs,
    typename... Extras
>
RPP_KERNEL void block_kernel(
    const BasisPack<Bases...> bases,
    const std::tuple<BatchArgs...> batches,
    const typename Op::Index batch_size,
    const Extras... extras
) {
    using Strategy = typename Op::Strategy;

    static_assert(strategies::is_block_strategy_v<Strategy>,
                  "this kernel is for block strategies only");


    extern __shared__ std::byte smem_bytes[];

    const auto ctx = Strategy::make_context(smem_bytes);

    Op::init_scratch_mem(ctx, bases);

    const auto work_idx = Strategy::object_index(blockIdx.x, threadIdx.x);
    if (work_idx >= batch_size) { return; }

    detail::invoke(Op{}, ctx, [&](auto const &batch) { return batch.view(work_idx, bases); }, batches, extras...);

    Op::destroy_scratch_mem(ctx, bases);
}
} // namespace rpp::gpu::block

#endif //RPP_GPU_OPERATIONS_BLOCK_KERNEL_HPP
