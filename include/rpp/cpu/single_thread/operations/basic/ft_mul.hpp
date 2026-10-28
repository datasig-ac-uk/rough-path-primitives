#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_MUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_MUL_HPP

// IWYU pragma: always_keep

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_mul.hpp>

#include <rpp/views/batch.hpp>

// #include <rpp/cpu/single_thread/operations/basic/ft_inplace_fma.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>

#include <rpp/cpu/single_thread/operations/basic/detail/ft_multiply.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class FTMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    static constexpr bool is_implemented = true;

    template <typename Basis>
    static constexpr std::size_t
    scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorLhs const& lhs,
                    TensorRhs const& rhs,
                    Accum beta = Accum{1}) const noexcept {
        namespace ftm = cpu::single_thread::ft_multiply;

        ftm::multiply(ctx,
                      ftm::detail::select_tile_config<4>(ctx, out.basis()),
                      out,
                      lhs,
                      rhs,
                      ftm::ScaledTileWriter<Context>{beta});
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_MUL_HPP
