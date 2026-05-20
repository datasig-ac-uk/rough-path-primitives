#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_MUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_MUL_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>
#include <rpp/operations/basic/ft_mul.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_inplace_fma.hpp>

namespace rpp::ops {
//
// template <typename Accum_, typename Architecture>
// class FTMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
//     using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
//     using Context = typename Strategy::Context;
//     using Accum = typename Strategy::Accum;
//
//     FTFma<Strategy> fma;
//
//
// public:
//     template <typename Basis>
//     static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
//         ignore_unused(strategy, basis);
//         return 0;
//     }
//
//     template <typename TensorOut, typename TensorLhs, typename TensorRhs>
//     void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs const& rhs, Accum beta = Accum{1}) const noexcept {
//         fma(ctx, out, out, lhs, rhs, Accum{0}, beta);
//     }
// };

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_MUL_HPP