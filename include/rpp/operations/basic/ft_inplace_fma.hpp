#ifndef RPP_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP
#define RPP_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

enum class FTInplaceFMAType {
    AEqualsBCPlusA, // a <- b*c + a
    AEqualsABPlusC, // a <- a*b + c
    AEqualsBAPlusC  // a <- b*a + c
};

template <typename Strategy, FTInplaceFMAType FMAType, typename=void>
class FTInplaceFma : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;

    using Accum = typename Strategy::Accum;
public:
    template <typename TensorA, typename TensorB, typename TensorC>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorA& a, TensorB const& b, TensorC const& c, Accum alpha=Accum{1}, Accum beta=Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorA, TensorB, TensorC, Accum>,
            "rpp::ops::FTInplaceFma has no implementation for this Strategy/FMA type. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy>
using FTInplaceFma231 = FTInplaceFma<Strategy, FTInplaceFMAType::AEqualsBCPlusA>;

template <typename Strategy>
using FTInplaceFma123 = FTInplaceFma<Strategy, FTInplaceFMAType::AEqualsABPlusC>;

template <typename Strategy>
using FTInplaceFma213 = FTInplaceFma<Strategy, FTInplaceFMAType::AEqualsABPlusC>;



} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP
