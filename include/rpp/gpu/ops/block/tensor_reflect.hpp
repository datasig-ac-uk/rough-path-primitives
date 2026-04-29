#ifndef RPP_GPU_OPS_BLOCK_TENSOR_REFLECT_HPP
#define RPP_GPU_OPS_BLOCK_TENSOR_REFLECT_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/utility.hpp>

#include <rpp/gpu/ops/block/detail/antipode.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class TensorReflect<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        gpu::block::generalised_antipode(ctx, out, arg, gpu::block::NoSigningPolicy{});
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void tensor_reflect_kernel(
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

    ops::TensorReflect<Strategy> op;

    op(ctx, batch_out.view(my_index, basis), batch_arg.view(my_index, basis));
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_TENSOR_REFLECT_HPP
