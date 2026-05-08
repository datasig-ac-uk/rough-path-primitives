#ifndef RPP_OPERATIONS_INTERMEDIATE_FT_EXP_HPP
#define RPP_OPERATIONS_INTERMEDIATE_FT_EXP_HPP

#include <cstddef>
#include <algorithm>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/ft_inplace_mul.hpp>
#include <rpp/operations/basic/tensor_add_identity.hpp>
#include <rpp/operations/basic/tensor_set_identity.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class FTExp {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using InplaceMul = FTInplaceMul<Strategy>;
    using SetIdentity = TensorSetIdentity<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    InplaceMul inplace_mul;
    SetIdentity set_identity;
    AddIdentity add_identity;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return std::max(InplaceMul::scratch_space_size(strategy, basis),
            std::max(SetIdentity::scratch_space_size(strategy, basis),
                        AddIdentity::scratch_space_size(strategy, basis)));
    }

    template <typename TensorOut, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        const Accum one { 1 };

        set_identity(ctx, out);

        for (Degree d = basis.depth; d > 0; --d) {
            const Accum divisor = one / d;

            ctx.sync();

            inplace_mul(ctx, out, arg.truncate(1, basis.depth), divisor);

            ctx.sync();
            add_identity(ctx, out);
        }
    }
};

} // namespace rpp::ops


#endif //RPP_OPERATIONS_INTERMEDIATE_FT_EXP_HPP
