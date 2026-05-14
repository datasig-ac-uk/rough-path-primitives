#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_SET_CONSTANT_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_SET_CONSTANT_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/vector_set_constant.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class VectorSetConstant<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    static constexpr bool is_implemented = true;

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

            for (Index i=ctx.thread_rank(); i<size; i += ctx.num_threads()) { data[i] = value;
            }
        } else {
            for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
                data[i] = value;
            }
        }

    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchVector, typename Basis, typename Value, typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void vector_set_constant_kernel(
    const BatchVector batch_vec,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    const Value value
) {
    using Strategy = strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::VectorSetConstant<Strategy> op;

    auto vec = batch_vec.view(my_index, basis);
    op(ctx, vec, value);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_SET_CONSTANT_HPP
