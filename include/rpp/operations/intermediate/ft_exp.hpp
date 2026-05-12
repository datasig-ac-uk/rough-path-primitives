#ifndef RPP_OPERATIONS_INTERMEDIATE_FT_EXP_HPP
#define RPP_OPERATIONS_INTERMEDIATE_FT_EXP_HPP

#include <cstddef>
#include <tuple>
#include <utility>
#include <algorithm>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_inplace_mul.hpp>
#include <rpp/operations/basic/tensor_add_identity.hpp>
#include <rpp/operations/basic/tensor_set_identity.hpp>

namespace rpp::ops {
template<typename Strategy, typename=void>
class FTExp : public BaseOperation<Strategy> {
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
    static constexpr bool is_implemented = InplaceMul::is_implemented && SetIdentity::is_implemented &&
                                           AddIdentity::is_implemented;

    template<typename Basis>
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        return std::max(InplaceMul::scratch_space_size(strategy, basis),
                        std::max(SetIdentity::scratch_space_size(strategy, basis),
                                 AddIdentity::scratch_space_size(strategy, basis)));
    }

    template<typename TensorOut, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const &ctx, TensorOut &out, TensorArg const &arg) const noexcept {
        auto const &basis = out.basis();
        const Accum one{1};

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

template <typename Strategy, typename BatchOut, typename BatchArg, typename Basis>
auto ft_exp(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchArg const& arg,
    Basis const& basis,
    typename Strategy::Index batch_size
    ) noexcept {
    using Op = FTExp<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"FTExp\" that implements \"ft_exp\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/intermediate/ft_exp.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, arg),
        basis,
        batch_size
        );
}
} // namespace rpp::ops


#endif //RPP_OPERATIONS_INTERMEDIATE_FT_EXP_HPP
