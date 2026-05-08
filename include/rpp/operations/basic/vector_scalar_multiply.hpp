#ifndef RPP_OPERATIONS_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
#define RPP_OPERATIONS_BASIC_VECTOR_SCALAR_MULTIPLY_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class VectorScalarMultiply : public BaseOperation<Strategy> {
    using Accum = typename Strategy::Accum;
public:
    using Context = typename Strategy::Context;

    template<typename Vector>
    RPP_HOST_DEVICE
    void operator()(Context const &ctx, Vector &vec, Accum scalar) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, Vector, Accum>,
            "rpp::ops::VectorScalarMultiply has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

} // namespace rpp::ops


#endif //RPP_OPERATIONS_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
