#ifndef RPP_OPERATIONS_INTERMEDIATE_FT_LOG_HPP
#define RPP_OPERATIONS_INTERMEDIATE_FT_LOG_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/ft_mul.hpp>
#include <rpp/operations/basic/tensor_add_identity.hpp>
#include <rpp/operations/basic/vector_set_constant.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class FTLog {
    using Context = typename Strategy::Context;

    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using SetConstant =VectorSetConstant<Strategy>;
    using InplaceMul = FTInplaceMul<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    SetConstant set_constant;
    InplaceMul inplace_mul;
    AddIdentity add_identity;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return std::max(
            SetConstant::scratch_space_size(strategy, basis),
            std::max(InplaceMul::scratch_space_size(strategy, basis),
                AddIdentity::scratch_space_size(strategy, basis))
            );

    }

    template <typename TensorOut, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        const Accum one { 1 };

        set_constant(ctx, out, Accum{0});

        for (Degree d=basis.depth; d > 0; --d) {
            const auto max_depth = basis.depth - d + 1;
            const Accum val = (d % 2 == 0 ? -one : one) / d;

            add_identity(ctx, out, val);

            ctx.sync();
            auto out_trunc = out.truncate(0, max_depth);
            inplace_mul(ctx, out_trunc, arg.truncate(1, max_depth));
        }
    }
};


} // namespace rpp::ops

#endif //RPP_OPERATIONS_INTERMEDIATE_FT_LOG_HPP
