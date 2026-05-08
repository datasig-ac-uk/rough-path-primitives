#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_FT_INPLACE_FMA_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_FT_INPLACE_FMA_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_inplace_fma.hpp>

#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/operations/block/basic/detail/ft_multiply.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture, FTInplaceFMAType FMAType>
class FTInplaceFma<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>, FMAType> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:

    template <typename TensorA, typename TensorB, typename TensorC>
    RPP_DEVICE void operator()(
        Context const& ctx,
        TensorA& a,
        TensorB const& b,
        TensorC const& c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) const noexcept {
        using Scalar = typename TensorA::value_type;
        auto const& basis = a.basis();
        const auto low_range_degree = std::min(a.max_degree(), ctx.low_range_degree(basis));

        for (Degree out_deg = a.max_degree(); out_deg > low_range_degree; --out_deg) {
            for (auto elt_idx = basis.start_of_degree(out_deg) + ctx.thread_rank(); elt_idx < basis.end_of_degree(out_deg); elt_idx += ctx.num_threads()) {
                if constexpr (FMAType == FTInplaceFMAType::AEqualsABPlusC) {
                    auto acc = gpu::block::ft_multiply_loop_with_degree(ctx, a, b, elt_idx, out_deg, basis);
                    acc *= beta;
                    Accum c_val{0};
                    if (c.has_degree(out_deg)) {
                        c_val = Accum{c[elt_idx]};
                    }
                    a[elt_idx] = static_cast<Scalar>(alpha * c_val + acc);
                } else if constexpr (FMAType == FTInplaceFMAType::AEqualsBAPlusC) {
                    auto acc = gpu::block::ft_multiply_loop_with_degree(ctx, b, a, elt_idx, out_deg, basis);
                    acc *= beta;
                    Accum c_val{0};
                    if (c.has_degree(out_deg)) {
                        c_val = Accum{c[elt_idx]};
                    }
                    a[elt_idx] = static_cast<Scalar>(alpha * c_val + acc);
                } else if constexpr (FMAType == FTInplaceFMAType::AEqualsBCPlusA) {
                    auto acc = gpu::block::ft_multiply_loop_with_degree(ctx, b, c, elt_idx, out_deg, basis);
                    acc *= beta;
                    a[elt_idx] = static_cast<Scalar>(alpha * Accum{a[elt_idx]} + acc);
                } else {
                    RPP_UNREACHABLE();
                }
            }
            ctx.sync();
        }

        auto elt_idx = static_cast<Index>(ctx.thread_rank());
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);
        Accum acc{0};
        if (active) {
            const auto degree = basis.degree_linear(elt_idx);
            if constexpr (FMAType == FTInplaceFMAType::AEqualsABPlusC) {
                acc = gpu::block::ft_multiply_loop_with_degree(ctx, a, b, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum{c[elt_idx]};
            } else if constexpr (FMAType == FTInplaceFMAType::AEqualsBAPlusC) {
                acc = gpu::block::ft_multiply_loop_with_degree(ctx, b, a, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum{c[elt_idx]};
            } else if constexpr (FMAType == FTInplaceFMAType::AEqualsBCPlusA) {
                acc = gpu::block::ft_multiply_loop_with_degree(ctx, b, c, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum{a[elt_idx]};
            } else {
                RPP_UNREACHABLE();
            }
        }

        ctx.sync();
        if (active) {
            a[elt_idx] = static_cast<Scalar>(acc);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <ops::FTInplaceFMAType FMAType, typename BatchA, typename BatchB, typename BatchC, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void ft_inplace_fma_kernel(
    const BatchA batch_a,
    const BatchB batch_b,
    const BatchC batch_c,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1},
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::FTInplaceFma<Strategy, FMAType> op;

    auto a = batch_a.view(my_index, basis);
    auto b = batch_b.view(my_index, basis);
    auto c = batch_c.view(my_index, basis);
    op(ctx, a, b, c, alpha, beta);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_FT_INPLACE_FMA_HPP
