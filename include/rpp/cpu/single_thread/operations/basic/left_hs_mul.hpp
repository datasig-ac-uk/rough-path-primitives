//
// Created by sam on 30/09/2026.
//

#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_MUL_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/left_hs_mul.hpp>
#include <rpp/views/batch.hpp>


#include <rpp/cpu/single_thread/strategy.hpp>


namespace rpp::ops {

template <typename AccumT, typename ArchiectureT>
class LeftHSMul<cpu::strategies::SingleThreadStrategy<AccumT, ArchiectureT>>
    : public BaseOperation<cpu::strategies::SingleThreadStrategy<AccumT, ArchiectureT>>
{
    using Strategy = cpu::strategies::SingleThreadStrategy<AccumT, ArchiectureT>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

public:
    static constexpr bool is_implemented = true;

    template <typename TensorOut,
              typename TensorIntegrand,
              typename TensorIntegrator>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorIntegrand const& integrand,
                    TensorIntegrator const& integrator,
                    Accum beta = Accum{1}) const noexcept {


    }


};


} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
