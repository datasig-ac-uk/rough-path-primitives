#ifndef RPP_GPU_OPS_BLOCK_VECTOR_SET_ZERO_HPP
#define RPP_GPU_OPS_BLOCK_VECTOR_SET_ZERO_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class VectorSetZero<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {

public:

    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;


    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename Vector>
    RPP_DEVICE void operator()(Context const& ctx, Vector& vec) const noexcept {
        auto const& basis = vec.basis();
        const auto begin = basis.start_of_degree(vec.min_degree());
        auto size = basis.end_of_degree(vec.max_degree()) - begin;

        auto data = vec.data() + begin;
        if constexpr (std::is_pointer_v<decltype(data)>) {
            const auto count_to_align = static_cast<Index>(
                (Architecture::sector_alignment -
                reinterpret_cast<std::uintptr_t>(data) & (Architecture::sector_alignment - 1)) / sizeof(*data));

            for (Index i=ctx.thread_rank(); i < std::min(count_to_align, size); i += ctx.num_threads()) {
                data[i] = 0;
            }
            data += count_to_align;
            size -= count_to_align;

            for (Index i=ctx.thread_rank(); i<size; i += ctx.num_threads()) {
                data[i] = 0;
            }
        } else {
            for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
                data[i] = 0;
            }
        }

    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchVector, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void vector_set_zero_kernel(
    const BatchVector batch_vec,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::VectorSetZero<Strategy> op;

    op(ctx, batch_vec.view(my_index, basis));
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_VECTOR_SET_ZERO_HPP
