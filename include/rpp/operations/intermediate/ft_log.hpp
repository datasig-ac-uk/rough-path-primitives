#ifndef RPP_OPERATIONS_INTERMEDIATE_FT_LOG_HPP
#define RPP_OPERATIONS_INTERMEDIATE_FT_LOG_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_inplace_mul.hpp>
#include <rpp/operations/basic/ft_mul.hpp>
#include <rpp/operations/basic/tensor_add_identity.hpp>
#include <rpp/operations/linalg/vector_set_constant.hpp>

namespace rpp::ops {
template<typename Strategy, typename=void>
class FTLog : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;

    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using SetConstant = VectorSetConstant<Strategy>;
    using InplaceMul = rpp::ops::FTInplaceMul<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    SetConstant set_constant;
    InplaceMul inplace_mul;
    AddIdentity add_identity;

public:
    static constexpr bool is_implemented = InplaceMul::is_implemented && AddIdentity::is_implemented &&
                                           SetConstant::is_implemented;

    template<typename Basis>
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        return std::max(
            SetConstant::scratch_space_size(strategy, basis),
            std::max(InplaceMul::scratch_space_size(strategy, basis),
                     AddIdentity::scratch_space_size(strategy, basis))
        );
    }

    template<typename TensorOut, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const &ctx, TensorOut &out, TensorArg const &arg) const noexcept {
        auto const &basis = out.basis();
        const Accum one{1};

        set_constant(ctx, out, Accum{0});

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_depth = basis.depth - d + 1;
            const Accum val = (d % 2 == 0 ? -one : one) / d;

            add_identity(ctx, out, val);

            ctx.sync();
            auto out_trunc = out.truncate(0, max_depth);
            inplace_mul(ctx, out_trunc, arg.truncate(1, max_depth));
        }
    }
};

template <typename Strategy, typename BatchOut, typename BatchArg, typename Basis>
auto ft_log(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchArg const& arg,
    Basis const& basis,
    typename Strategy::Index num_batches
    ) noexcept {
    using Op = FTLog<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"FTLog\" that implements \"ft_log\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/intermediate/ft_log.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, arg),
        make_basis_pack(basis),
        num_batches
        );
}
} // namespace rpp::ops

#endif //RPP_OPERATIONS_INTERMEDIATE_FT_LOG_HPP
