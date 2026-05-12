#ifndef RPP_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class TensorSetIdentity : public BaseOperation<Strategy> {
    using Accum = typename Strategy::Accum;
public:
    using Context = typename Strategy::Context;
    static constexpr bool is_implemented = false;


    template <typename Tensor>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, Tensor& tensor, Accum scalar=Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, Tensor, Accum>,
            "rpp::ops::TensorSetIdentity has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchTensor, typename Basis>
auto tensor_set_identity(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchTensor const& tensor,
    Basis const& basis,
    typename Strategy::Index batch_size,
    typename Strategy::Accum scalar = typename Strategy::Accum{1}
    ) noexcept {
    using Op = TensorSetIdentity<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"TensorSetIdentity\" that implements "
        "\"tensor_set_identity\" is not implemented. This either means that the "
        "Strategy object is invalid, or that the necessary specialisation headers "
        "have not been included. For example, you may need to add the following "
        "include directive to bring in the single-threaded CPU implementation of "
        "this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/tensor_set_identity.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(tensor),
        basis,
        batch_size,
        scalar
        );
}

} // namespace rpp::ops


#endif //RPP_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP
