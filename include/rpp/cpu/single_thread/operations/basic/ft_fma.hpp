#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_FMA_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_FMA_HPP

// IWYU pragma: always_keep

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/ft_fma.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>

#include <rpp/cpu/single_thread/operations/basic/detail/ft_multiply.hpp>


namespace rpp::ops {
template <typename Accum_, typename Architecture>
class FTFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public cpu::single_thread::ft_multiply::Operation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>,
          cpu::single_thread::ft_multiply::default_tile_power> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

public:
    static constexpr bool is_implemented = true;

    template <typename TensorOut,
              typename TensorA,
              typename TensorB,
              typename TensorC>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorA const& a,
                    TensorB const& b,
                    TensorC const& c,
                    Accum alpha = Accum{1},
                    Accum beta = Accum{1}) const noexcept {
        namespace ftm = cpu::single_thread::ft_multiply;

        ftm::multiply(
            ctx,
            ftm::detail::select_tile_config<ftm::default_tile_power>(
                ctx, out.basis()),
            out,
            b,
            c,
            ftm::ScaledTileAddWriter<Context, TensorA>{a, alpha, beta});
    }
};
} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_FMA_HPP
