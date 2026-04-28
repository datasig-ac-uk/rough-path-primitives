#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_EXP_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_EXP_HPP

#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>


#include <rpp/cpu/ops/single_thread/ft_inplace_mul.hpp>
#include <rpp/cpu/ops/single_thread/tensor_add_identity.hpp>
#include <rpp/cpu/ops/single_thread/tensor_add_identity.hpp>


namespace rpp::ops {

// template <typename Accum_, typename Architecture>
// class FTExp<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
//     using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
//     using Context = typename Strategy::Context;
//
// public:
//     template <typename LaunchConfig, typename Basis>
//     static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
//         ignore_unused(config, basis);
//         return 0;
//     }
//
//     template <typename TensorOut, typename TensorArg>
//     void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
//         ignore_unused(ctx, out, arg);
//     }
// };

} // namespace rpp::ops

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_EXP_HPP
