#ifndef RPP_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class TensorSetIdentity : public BaseOperation<Strategy> {
    using Accum = typename Strategy::Accum;
public:
    using Context = typename Strategy::Context;

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

} // namespace rpp::ops


#endif //RPP_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP
