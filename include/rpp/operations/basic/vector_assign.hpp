#ifndef RPP_OPERATIONS_BASIC_VECTOR_ASSIGN_HPP
#define RPP_OPERATIONS_BASIC_VECTOR_ASSIGN_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class VectorAssign : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
public:
    static constexpr bool is_implemented = false;

    template <typename VectorOut, typename VectorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, VectorOut& out, VectorArg const& arg) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, VectorOut, VectorArg>,
            "rpp::ops::VectorAssign has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

} // namespace rpp::ops


#endif //RPP_OPERATIONS_BASIC_VECTOR_ASSIGN_HPP
