#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_ASSIGN_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_ASSIGN_HPP

#include <algorithm>

#include <cuda/std/__ranges/data.h>
#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class VectorAssign<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Architecture_ = typename Strategy::Architecture;
    using Index = typename Strategy::Index;


    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }


    template <typename VectorOut, typename VectorArg>
    RPP_DEVICE void operator()(Context const& ctx, VectorOut& out, VectorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        const auto begin = basis.start_of_degree(std::max(out.min_degree(), arg.min_degree()));
        auto size = basis.end_of_degree(std::min(out.max_degree(), arg.max_degree())) - begin;

        auto arg_data = arg.data() + begin;
        auto out_data = out.data() + begin;
        if constexpr (std::is_pointer_v<decltype(arg_data)>) {
            const auto count_to_align = static_cast<Index>(
                (Architecture_::sector_alignment -
                (reinterpret_cast<std::uintptr_t>(arg_data) & (Architecture_::sector_alignment - 1))) / sizeof(*arg_data));

            for (Index i=ctx.thread_rank(); i < std::min(count_to_align, size); i += ctx.num_threads()) {
                out_data[i] = arg_data[i];
            }
            arg_data += count_to_align;
            out_data += count_to_align;
            size -= count_to_align;

            for (Index i=ctx.thread_rank(); i<size; i += ctx.num_threads()) {
                out_data[i] = arg_data[i];
            }
        } else {
            for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
                out_data[i] = arg_data[i];
            }
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void vector_assign_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::VectorAssign<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto arg = batch_arg.view(my_index, basis);
    op(ctx, out, arg);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_ASSIGN_HPP
