#ifndef RPP_GPU_BLOCK_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP
#define RPP_GPU_BLOCK_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/gpu/block/strategy.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/intermediate/ft_fmexp.hpp>

#include <rpp/gpu/block/operations/basic/ft_inplace_fma.hpp>
#include <rpp/gpu/block/operations/linalg/vector_assign.hpp>

namespace rpp::ops {

template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class FTFMExp<gpu::strategies::
                  BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

private:
    using InplaceFMA123 = FTInplaceFma123<Strategy>;
    using Assign = VectorAssign<Strategy>;

    InplaceFMA123 inplace_fma123;
    Assign assign;

public:
    static constexpr bool is_implemented = true;

    template <typename BasisPack>
    static constexpr size_t scratch_space_size(Strategy const& strategy,
                                               BasisPack const& pack) noexcept {
        return std::max(InplaceFMA123::scratch_space_size(strategy, pack),
                        Assign::scratch_space_size(strategy, pack));
    }

    template <typename TensorOut,
              typename TensorMultiplier,
              typename TensorExponent>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorMultiplier const& multiplier,
                               TensorExponent const& exponent) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one{1};
        assign(ctx, out, multiplier);

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;
            ctx.sync();
            inplace_fma123(ctx,
                           out,
                           exponent.truncate(1, max_degree),
                           multiplier,
                           one,
                           divisor);
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP
