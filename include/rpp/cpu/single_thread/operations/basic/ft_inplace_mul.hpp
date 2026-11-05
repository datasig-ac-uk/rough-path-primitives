#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP

// IWYU pragma: always_keep

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/ft_inplace_mul.hpp>

#include <rpp/cpu/single_thread/operations/basic/detail/ft_multiply.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class FTInplaceMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public cpu::single_thread::ft_multiply::Operation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>,
          cpu::single_thread::ft_multiply::default_tile_power> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using BaseOperation<Strategy>::BaseOperation;

    static constexpr bool is_implemented = true;

    template <typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx,
                    TensorLhs& lhs,
                    TensorRhs const& rhs,
                    Accum beta = Accum{1}) const noexcept {
        namespace ftm = cpu::single_thread::ft_multiply;

        ftm::multiply(ctx,
                      ftm::detail::select_tile_config<ftm::default_tile_power>(
                          ctx, lhs.basis()),
                      lhs,
                      lhs,
                      rhs,
                      ftm::ScaledTileWriter<Context>{beta});
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP
