#ifndef RPP_GPU_BLOCK_OPERATIONS_INTERMEDIATE_FT_LOG_HPP
#define RPP_GPU_BLOCK_OPERATIONS_INTERMEDIATE_FT_LOG_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/gpu/block/strategy.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/intermediate/ft_log.hpp>

#include <rpp/gpu/block/operations/basic/ft_inplace_mul.hpp>
#include <rpp/gpu/block/operations/basic/tensor_add_identity.hpp>
#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>

namespace rpp::ops {

template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class FTLog<gpu::strategies::
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
    using SetZero = VectorSetConstant<Strategy>;
    using InplaceMul = FTInplaceMul<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    SetZero set_zero;
    InplaceMul inplace_mul;
    AddIdentity add_identity;

public:
    static constexpr bool is_implemented = true;

    template <typename BasisPack>
    static constexpr size_t scratch_space_size(Strategy const& strategy,
                                               BasisPack const& pack) noexcept {
        return std::max(
            SetZero::scratch_space_size(strategy, pack),
            std::max(InplaceMul::scratch_space_size(strategy, pack),
                     AddIdentity::scratch_space_size(strategy, pack)));
    }

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one{1};
        set_zero(ctx, out, Accum{0});

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum val = (d % 2 == 0 ? -one : one) / d;
            add_identity(ctx, out, val);
            ctx.sync();
            auto trunc_out = out.truncate(0, max_degree);
            inplace_mul(ctx, trunc_out, arg.truncate(1, max_degree));
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_INTERMEDIATE_FT_LOG_HPP
