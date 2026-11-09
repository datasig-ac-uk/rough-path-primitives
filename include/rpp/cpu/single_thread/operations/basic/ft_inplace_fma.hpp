#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP

// IWYU pragma: always_keep

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/ft_inplace_fma.hpp>

#include <rpp/cpu/single_thread/operations/basic/detail/ft_multiply.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture, FTInplaceFMAType FMAType>
class FTInplaceFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>,
                   FMAType>
    : public cpu::single_thread::ft_multiply::Operation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>,
          cpu::single_thread::ft_multiply::default_tile_power> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

    using Base = cpu::single_thread::ft_multiply::Operation<
        Strategy,
        cpu::single_thread::ft_multiply::default_tile_power>;

public:
    using Base::Base;

    static constexpr bool is_implemented = true;

    template <typename TensorA, typename TensorB, typename TensorC>
    void operator()(Context const& ctx,
                    TensorA& a,
                    TensorB const& b,
                    TensorC const& c,
                    Accum alpha = Accum{1},
                    Accum beta = Accum{1}) const noexcept {
        namespace ftm = cpu::single_thread::ft_multiply;

        const auto config =
            ftm::detail::select_tile_config<ftm::default_tile_power>(
                ctx, a.basis());
        if constexpr (FMAType == FTInplaceFMAType::AEqualsBCPlusA) {
            using Writer = ftm::InplaceFmaTileWriter<Context, FMAType, TensorA>;
            ftm::multiply(ctx, config, a, b, c, Writer{alpha, beta});
        }
        else if constexpr (FMAType == FTInplaceFMAType::AEqualsABPlusC) {
            using Writer = ftm::InplaceFmaTileWriter<Context, FMAType, TensorC>;
            ftm::multiply(ctx, config, a, a, b, Writer{c, alpha, beta});
        }
        else {
            static_assert(FMAType == FTInplaceFMAType::AEqualsBAPlusC,
                          "unsupported in-place FMA type");
            using Writer = ftm::InplaceFmaTileWriter<Context, FMAType, TensorC>;
            ftm::multiply(ctx, config, a, b, a, Writer{c, alpha, beta});
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_FT_INPLACE_FMA_HPP
