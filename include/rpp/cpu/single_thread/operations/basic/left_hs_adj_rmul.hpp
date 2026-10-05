#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP

#include <rpp/cpu/single_thread/strategy.hpp>
#include <rpp/operations/basic/left_hs_adj_rmul.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class LeftHSAdjRMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
