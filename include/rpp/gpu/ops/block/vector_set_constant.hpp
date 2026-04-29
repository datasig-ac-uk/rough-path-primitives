#ifndef RPP_GPU_OPS_BLOCK_VECTOR_SET_CONSTANT_HPP
#define RPP_GPU_OPS_BLOCK_VECTOR_SET_CONSTANT_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class VectorSetConstant<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Index = typename Strategy::Index;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename Vector, typename Value>
    RPP_DEVICE void operator()(Context const& ctx, Vector& vec, Value const& value) const noexcept {
        using Scalar = typename Vector::value_type;
        auto const& basis = vec.basis();
        const auto begin = basis.start_of_degree(vec.min_degree());
        auto size = basis.end_of_degree(vec.max_degree()) - begin;

        auto data = vec.data() + begin;
        if constexpr (std::is_pointer_v<decltype(data)>) {
            const auto count_to_align = static_cast<Index>(
                (Architecture::sector_alignment -
                reinterpret_cast<std::uintptr_t>(data) & (Architecture::sector_alignment - 1)) / sizeof(*data));

            for (Index i=ctx.thread_rank(); i < std::min(count_to_align, size); i += ctx.num_threads()) {
                data[i] = value;
            }
            data += count_to_align;
            size -= count_to_align;

            for (Index i=ctx.thread_rank(); i<size; i += ctx.num_threads()) {
                data[i] = 0;
            }
        } else {
            for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
                data[i] = static_cast<Scalar>(value);
            }
        }

    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchVector, typename Basis, typename Value, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void vector_set_constant_kernel(
    const BatchVector batch_vec,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    const Value value
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::VectorSetConstant<Strategy> op;

    op(ctx, batch_vec.view(my_index, basis), value);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_VECTOR_SET_CONSTANT_HPP
