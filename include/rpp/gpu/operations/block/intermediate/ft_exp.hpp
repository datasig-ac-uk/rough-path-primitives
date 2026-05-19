#ifndef RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_EXP_HPP
#define RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_EXP_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/intermediate/ft_exp.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>
#include <rpp/gpu/operations/block/basic/ft_inplace_mul.hpp>
#include <rpp/gpu/operations/block/basic/tensor_add_identity.hpp>
#include <rpp/gpu/operations/block/basic/tensor_set_identity.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class FTExp<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

private:
    using InplaceMul = FTInplaceMul<Strategy>;
    using SetIdentity = TensorSetIdentity<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    InplaceMul inplace_mul;
    SetIdentity set_identity;
    AddIdentity add_identity;

public:
    static constexpr bool is_implemented = true;

    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return std::max(InplaceMul::scratch_space_size(strategy, basis),
                        std::max(SetIdentity::scratch_space_size(strategy, basis),
                                 AddIdentity::scratch_space_size(strategy, basis)));
    }

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one{1};
        set_identity(ctx, out);

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;
            ctx.sync();
            inplace_mul(ctx, out, arg.truncate(1, max_degree), divisor);
            add_identity(ctx, out);
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_EXP_HPP
